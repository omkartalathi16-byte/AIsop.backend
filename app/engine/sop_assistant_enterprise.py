"""
sop_assistant_enterprise.py
Enterprise SOP Assistant - Lightweight Hybrid Architecture

Architecture:
User Query → DeriveEngine → Hybrid Extractive Summarizer → 
Instruction Extraction → Dynamic Counter-Question Logic → NLP Response Formatter

Components:
- DeriveEngine: Your existing retrieval system (Milvus + metadata boosting)
- SentenceTransformer: Only used for sentence ranking in summarization
- NLTK + spaCy: Rule-based NLP for extraction
- LangGraph: Conversation orchestration
"""

import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# SentenceTransformer - ONLY for sentence similarity scoring
from sentence_transformers import SentenceTransformer

# NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# spaCy for advanced NLP
import spacy
from spacy.cli import download

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

import logging
logger = logging.getLogger(__name__)


##############################################################################
# CONFIGURATION
##############################################################################

@dataclass
class SOPAssistantConfig:
    """Lightweight configuration for SOP Assistant."""
    
    # DeriveEngine integration
    derive_top_k: int = 3
    derive_threat_type: Optional[str] = None
    derive_category: Optional[str] = None
    
    # SentenceTransformer - ONLY model loaded
    sentence_transformer_model: str = "all-MiniLM-L6-v2"  # 384-dim, fast
    
    # Summarization
    max_summary_sentences: int = 5
    summary_similarity_threshold: float = 0.5
    enable_position_bias: bool = True
    position_bias_weight: float = 0.3
    
    # Instruction extraction
    enable_instruction_extraction: bool = True
    max_instruction_steps: int = 15
    
    # Counter-question logic
    enable_counter_questions: bool = True
    max_counter_questions: int = 3
    
    # Conversation
    max_conversation_turns: int = 20
    enable_persistence: bool = True
    
    # NLP
    language: str = 'english'
    enable_lemmatization: bool = True


##############################################################################
# INTENT CLASSIFIER (Rule-based)
##############################################################################

class IntentType(Enum):
    """User intent types."""
    NEW_SOP_QUERY = "new_sop_query"      # Looking for new SOP
    SOP_SUMMARY = "sop_summary"          # Summarize current SOP
    INSTRUCTION = "instruction"          # Step-by-step instructions
    CLARIFICATION = "clarification"      # Follow-up on current SOP
    COMPARISON = "comparison"           # Compare SOPs
    COUNTER_QUESTION = "counter_question" # Response to assistant's question
    GREETING = "greeting"               # Hello, hi, etc.
    THANKS = "thanks"                  # Thank you
    UNKNOWN = "unknown"


class IntentClassifier:
    """Rule-based intent classification - no ML, no LLM."""
    
    def __init__(self):
        # Greeting patterns
        self.greeting_patterns = [
            r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)',
            r'^howdy',
            r'^what\'?s up'
        ]
        
        # Thanks patterns
        self.thanks_patterns = [
            r'thanks?',
            r'thank you',
            r'appreciate it',
            r'that helps?',
            r'got it'
        ]
        
        # Summary patterns
        self.summary_patterns = [
            r'summarize',
            r'summary',
            r'overview',
            r'brie[f|v]',
            r'tell me about this',
            r'explain this',
            r'what (is|are) this about',
            r'give me the (key|main) points'
        ]
        
        # Instruction patterns
        self.instruction_patterns = [
            r'step',
            r'how (do|can) I',
            r'instructions?',
            r'procedure',
            r'guide',
            r'walk me through',
            r'what (do|should) I (do|click|press|enter)',
            r'next step',
            r'follow'
        ]
        
        # Clarification patterns
        self.clarification_patterns = [
            r'what (do|does) (that|this|it) mean',
            r'explain (that|this)',
            r'can you elaborate',
            r'tell me more',
            r'clarify',
            r'what about',
            r'and then',
            r'after that',
            r'specifically',
            r'for example'
        ]
        
        # Comparison patterns
        self.comparison_patterns = [
            r'compare',
            r'difference',
            r'versus',
            r'vs',
            r'better',
            r'which one',
            r'what\'?s the difference'
        ]
        
        # Counter question response patterns
        self.counter_question_patterns = [
            r'^yes',
            r'^yeah',
            r'^yep',
            r'^sure',
            r'^ok',
            r'^okay',
            r'^please do',
            r'^that would be great'
        ]
        
        # New SOP query patterns
        self.new_sop_patterns = [
            r'how (do|can) I',
            r'what is',
            r'where (do|can) I',
            r'tell me about',
            r'show me',
            r'find',
            r'search',
            r'looking for',
            r'need help with',
            r'have a question about'
        ]
    
    def classify(self, query: str, has_active_sop: bool = False, 
                 awaiting_counter_response: bool = False) -> IntentType:
        """Classify user intent using pattern matching."""
        query_lower = query.lower().strip()
        
        # Check for empty query
        if not query_lower:
            return IntentType.UNKNOWN
        
        # Priority 1: Check if responding to counter-question
        if awaiting_counter_response:
            for pattern in self.counter_question_patterns:
                if re.match(pattern, query_lower):
                    return IntentType.COUNTER_QUESTION
        
        # Priority 2: Check greetings
        for pattern in self.greeting_patterns:
            if re.match(pattern, query_lower):
                return IntentType.GREETING
        
        # Priority 3: Check thanks
        for pattern in self.thanks_patterns:
            if re.search(pattern, query_lower):
                return IntentType.THANKS
        
        # Priority 4: Check summary requests
        for pattern in self.summary_patterns:
            if re.search(pattern, query_lower):
                if has_active_sop:
                    return IntentType.SOP_SUMMARY
                else:
                    return IntentType.NEW_SOP_QUERY
        
        # Priority 5: Check instruction requests
        for pattern in self.instruction_patterns:
            if re.search(pattern, query_lower):
                if has_active_sop:
                    return IntentType.INSTRUCTION
                else:
                    return IntentType.NEW_SOP_QUERY
        
        # Priority 6: Check clarification
        for pattern in self.clarification_patterns:
            if re.search(pattern, query_lower):
                if has_active_sop:
                    return IntentType.CLARIFICATION
        
        # Priority 7: Check comparison
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return IntentType.COMPARISON
        
        # Priority 8: Check if it's a new SOP query
        if not has_active_sop:
            for pattern in self.new_sop_patterns:
                if re.search(pattern, query_lower):
                    return IntentType.NEW_SOP_QUERY
        
        # Default: If has active SOP, assume clarification
        if has_active_sop:
            return IntentType.CLARIFICATION
        
        return IntentType.NEW_SOP_QUERY


##############################################################################
# HYBRID SUMMARIZER - SentenceTransformer + NLP
##############################################################################

class SentenceEmbeddingRanker:
    """
    Lightweight sentence ranking using SentenceTransformer.
    Only model loaded - no FAISS, no vector stores.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", transformer_model: Any = None):
        """Initialize SentenceTransformer model."""
        if transformer_model:
            self.model = transformer_model
            logger.info("Using shared SentenceTransformer model instance")
        else:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model.eval()  # Inference mode
            logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings."""
        if not sentences:
            return np.array([])
        return self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    
    def rank_sentences(self, sentences: List[str], reference: str = None) -> List[Tuple[int, float]]:
        """
        Rank sentences by relevance to reference or by importance to document.
        
        Args:
            sentences: List of sentences to rank
            reference: Reference text (query or document)
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        if not sentences:
            return []
        
        # Encode all sentences
        sent_embeddings = self.encode(sentences)
        
        if reference and len(sentences) > 1:
            # Query-focused ranking
            ref_embedding = self.encode([reference])[0]
            scores = np.dot(sent_embeddings, ref_embedding)
        else:
            # Document-focused ranking - use centroid
            centroid = np.mean(sent_embeddings, axis=0)
            scores = np.dot(sent_embeddings, centroid)
        
        # Create list of (index, score) tuples
        ranked = [(i, float(score)) for i, score in enumerate(scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked


class SOPExtractiveSummarizer:
    """
    Pure extractive summarizer - NO generative models.
    Combines SentenceTransformer ranking with NLP rules.
    """
    
    def __init__(self, config: SOPAssistantConfig, transformer_model: Any = None):
        self.config = config
        self.ranker = SentenceEmbeddingRanker(
            config.sentence_transformer_model, 
            transformer_model=transformer_model
        )
        
        # NLP tools
        self.stopwords = set(stopwords.words(config.language))
        self.lemmatizer = WordNetLemmatizer() if config.enable_lemmatization else None
        
        # Section markers for SOPs
        self.section_markers = {
            'purpose': ['purpose', 'objective', 'goal', 'overview', 'summary'],
            'scope': ['scope', 'applicability', 'who', 'audience'],
            'prerequisites': ['prerequisite', 'requirement', 'before', 'need', 'must have'],
            'procedure': ['procedure', 'step', 'instruction', 'how to', 'process'],
            'warning': ['warning', 'caution', 'important', 'note', 'critical', 'alert'],
            'contact': ['contact', 'support', 'help desk', 'escalation', 'report']
        }
        
        # Action verbs for instruction extraction
        self.action_verbs = {
            'click', 'select', 'enter', 'type', 'press', 'navigate', 'open',
            'close', 'save', 'delete', 'copy', 'paste', 'cut', 'drag',
            'drop', 'scroll', 'highlight', 'check', 'uncheck', 'enable',
            'disable', 'configure', 'set', 'update', 'modify', 'change',
            'add', 'remove', 'insert', 'create', 'generate', 'export',
            'import', 'download', 'upload', 'submit', 'verify', 'confirm',
            'review', 'approve', 'reject', 'escalate', 'notify', 'report',
            'run', 'execute', 'start', 'stop', 'restart', 'refresh',
            'connect', 'disconnect', 'install', 'uninstall', 'reset'
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Split text into sentences and clean."""
        if not text:
            return []
        
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
        # Clean and filter
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            # Filter very short sentences and pure numbers
            if len(sent) > 20 and not sent.isdigit():
                cleaned.append(sent)
        
        return cleaned
    
    def _identify_section(self, sentence: str) -> Optional[str]:
        """Identify which SOP section a sentence belongs to."""
        sent_lower = sentence.lower()
        for section, markers in self.section_markers.items():
            for marker in markers:
                if marker in sent_lower[:50]:  # Check beginning of sentence
                    return section
        return None
    
    def _apply_position_bias(self, scores: List[float], indices: List[int]) -> List[float]:
        """Apply position bias - earlier sentences get boost."""
        if not self.config.enable_position_bias:
            return scores
        
        biased_scores = []
        max_idx = max(indices) if indices else 1
        
        for idx, score in zip(indices, scores):
            # Linear position bias: earlier = higher boost
            position_boost = 1.0 - (idx / max_idx) * self.config.position_bias_weight
            biased_scores.append(score * position_boost)
        
        return biased_scores
    
    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using POS patterns."""
        doc = nlp(text[:5000])  # Limit text length
        
        # Extract noun chunks and verb phrases
        phrases = []
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) <= 3 and phrase not in self.stopwords:
                phrases.append(phrase)
        
        # Verb phrases (action + object)
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = child.text
                        break
                
                if obj:
                    phrases.append(f"{token.lemma_} {obj}".lower())
        
        # Count frequencies
        counter = Counter(phrases)
        
        # Return top phrases
        return [p for p, _ in counter.most_common(top_n)]
    
    def summarize(self, 
                  sop_chunks: List[Dict[str, Any]], 
                  query: str = None,
                  summary_type: str = "general") -> Dict[str, Any]:
        """
        Generate extractive summary using SentenceTransformer ranking.
        
        Args:
            sop_chunks: List of SOP chunks from DeriveEngine
            query: Optional query for query-focused summary
            summary_type: general, quick, detailed, or query-focused
            
        Returns:
            Summary dictionary with sentences and metadata
        """
        if not sop_chunks:
            return {
                "summary": "No content available.",
                "key_points": [],
                "sections": {},
                "sentence_count": 0
            }
        
        # Extract all content from chunks
        full_text = ' '.join([chunk.get('content', '') for chunk in sop_chunks])
        sentences = self._preprocess_text(full_text)
        
        if not sentences:
            return {
                "summary": full_text[:500] if full_text else "No content.",
                "key_points": [full_text[:200]] if full_text else [],
                "sentence_count": 0
            }
        
        # Rank sentences using SentenceTransformer
        reference = query if summary_type == "query-focused" else None
        ranked_sentences = self.ranker.rank_sentences(sentences, reference)
        
        # Extract scores and indices
        indices = [idx for idx, _ in ranked_sentences]
        scores = [score for _, score in ranked_sentences]
        
        # Apply position bias
        if summary_type != "query-focused":
            scores = self._apply_position_bias(scores, indices)
        
        # Re-rank with bias
        biased_ranked = list(zip(indices, scores))
        biased_ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on summary type
        if summary_type == "quick":
            num_sentences = min(3, len(sentences))
        elif summary_type == "detailed":
            num_sentences = min(self.config.max_summary_sentences, len(sentences))
        else:  # general or query-focused
            num_sentences = min(5, len(sentences))
        
        top_indices = [idx for idx, _ in biased_ranked[:num_sentences]]
        top_indices.sort()  # Restore original order
        
        # Build summary
        summary_sentences = [sentences[i] for i in top_indices]
        
        # Group by section
        sections = defaultdict(list)
        for i in top_indices:
            section = self._identify_section(sentences[i])
            if section:
                sections[section].append(sentences[i])
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(full_text)
        
        return {
            "summary": ' '.join(summary_sentences),
            "summary_sentences": summary_sentences,
            "key_points": summary_sentences[:3],
            "key_phrases": key_phrases[:8],
            "sections": dict(sections),
            "sentence_count": len(sentences),
            "selected_count": len(summary_sentences),
            "summary_type": summary_type,
            "confidence": float(np.mean(scores[:num_sentences])) if scores else 0.0
        }
    
    def extract_instructions(self, sop_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract step-by-step instructions using NLP rules.
        No neural networks - pure rule-based extraction.
        """
        instructions = []
        
        for chunk in sop_chunks:
            content = chunk.get('content', '')
            
            # Use spaCy for advanced pattern matching
            doc = nlp(content[:2000])  # Limit per chunk
            
            # ===== PATTERN 1: Numbered steps =====
            numbered_pattern = r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\n\s*[•\-*]|\Z)'
            numbered_matches = re.finditer(numbered_pattern, content, re.DOTALL | re.MULTILINE)
            
            for match in numbered_matches:
                step_num, step_text = match.groups()
                instructions.append({
                    "type": "numbered",
                    "step": int(step_num),
                    "text": step_text.strip(),
                    "confidence": 0.95
                })
            
            # ===== PATTERN 2: Bullet points =====
            bullet_pattern = r'(?:^|\n)\s*[•\-*]\s+(.+?)(?=\n\s*[•\-*]|\n\s*\d+[\.\)]|\Z)'
            bullet_matches = re.finditer(bullet_pattern, content, re.DOTALL | re.MULTILINE)
            
            for match in bullet_matches:
                bullet_text = match.group(1).strip()
                instructions.append({
                    "type": "bullet",
                    "text": bullet_text,
                    "confidence": 0.9
                })
            
            # ===== PATTERN 3: Imperative sentences =====
            for sent in doc.sents:
                sent_text = sent.text.strip()
                
                # Skip short sentences
                if len(sent_text) < 15:
                    continue
                
                # Check if starts with action verb
                first_token = sent[0]
                if first_token.pos_ == 'VERB' and first_token.dep_ == 'ROOT':
                    instructions.append({
                        "type": "action",
                        "verb": first_token.lemma_,
                        "text": sent_text,
                        "confidence": 0.85
                    })
                
                # Check for "Step X:" pattern
                step_pattern = re.match(r'^(?:step|phase|stage)\s+(\d+)[:\-]\s*(.+)$', sent_text, re.I)
                if step_pattern:
                    step_num, step_text = step_pattern.groups()
                    instructions.append({
                        "type": "numbered",
                        "step": int(step_num),
                        "text": step_text.strip(),
                        "confidence": 0.9
                    })
        
        # Deduplicate
        seen_texts = set()
        unique_instructions = []
        
        for inst in instructions:
            text_hash = hashlib.md5(inst['text'].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_instructions.append(inst)
        
        # Sort numbered steps
        numbered = [i for i in unique_instructions if i['type'] == 'numbered']
        numbered.sort(key=lambda x: x.get('step', 0))
        
        # Combine all, numbered first
        result = numbered + [i for i in unique_instructions if i['type'] != 'numbered']
        
        return result[:self.config.max_instruction_steps]
    
    def answer_question(self, question: str, sop_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Answer specific questions using SentenceTransformer similarity.
        No vector database - uses on-the-fly encoding.
        """
        if not sop_chunks:
            return {
                "answer": "No information available.",
                "confidence": 0.0,
                "source": None
            }
        
        # Extract all sentences from chunks
        all_sentences = []
        sentence_to_chunk = []
        
        for chunk in sop_chunks:
            content = chunk.get('content', '')
            sentences = sent_tokenize(content)
            
            for sent in sentences:
                if len(sent) > 20:
                    all_sentences.append(sent)
                    sentence_to_chunk.append(chunk)
        
        if not all_sentences:
            return {
                "answer": "No detailed information available.",
                "confidence": 0.0
            }
        
        # Rank sentences by similarity to question
        ranked = self.ranker.rank_sentences(all_sentences, question)
        
        if not ranked:
            return {
                "answer": "I couldn't find a specific answer.",
                "confidence": 0.0
            }
        
        # Get top sentence
        best_idx, best_score = ranked[0]
        
        if best_score < self.config.summary_similarity_threshold:
            return {
                "answer": "I couldn't find a confident answer to that question.",
                "confidence": best_score,
                "suggestions": self._generate_counter_questions(question, sop_chunks)
            }
        
        best_sentence = all_sentences[best_idx]
        source_chunk = sentence_to_chunk[best_idx]
        
        return {
            "answer": best_sentence,
            "confidence": best_score,
            "source": source_chunk.get('title', 'Unknown SOP'),
            "source_chunk": {
                "content": best_sentence,
                "similarity": best_score
            }
        }
    
    def _generate_counter_questions(self, question: str, sop_chunks: List[Dict]) -> List[str]:
        """Generate relevant counter-questions based on SOP content."""
        counter_questions = []
        
        # Extract key terms from question
        question_lower = question.lower()
        
        # Check for common question types
        if 'how' in question_lower:
            counter_questions.append("Would you like me to show you the step-by-step instructions?")
        
        if 'what' in question_lower:
            counter_questions.append("Would you like a summary of the key points?")
        
        if 'when' in question_lower or 'who' in question_lower:
            # Look for scope/prerequisites sections
            for chunk in sop_chunks[:3]:
                content = chunk.get('content', '').lower()
                if 'scope' in content or 'audience' in content:
                    counter_questions.append("I can also tell you who this SOP applies to.")
                    break
        
        # Add generic counter-question if none generated
        if not counter_questions and self.config.enable_counter_questions:
            counter_questions.append("Would you like more details about this procedure?")
        
        return counter_questions[:self.config.max_counter_questions]


##############################################################################
# RESPONSE FORMATTER
##############################################################################

class ResponseFormatter:
    """NLP-based response formatter - no templates."""
    
    @staticmethod
    def format_summary_response(sop_title: str, summary: Dict[str, Any], sop_link: str = None) -> str:
        """Format summary response with natural language."""
        lines = []
        
        # Header
        lines.append(f"📋 **{sop_title}**")
        lines.append("")
        
        # Key points
        if summary.get('key_points'):
            lines.append("**Key Points:**")
            for point in summary['key_points'][:3]:
                lines.append(f"• {point}")
            lines.append("")
        
        # Full summary
        if summary.get('summary'):
            lines.append("**Summary:**")
            lines.append(summary['summary'])
            lines.append("")
        
        # Sections if available
        if summary.get('sections'):
            sections = summary['sections']
            if 'purpose' in sections:
                lines.append("**Purpose:**")
                lines.append(sections['purpose'][0][:200] + "...")
                lines.append("")
        
        # Key phrases
        if summary.get('key_phrases'):
            lines.append("**Topics:** " + ", ".join(summary['key_phrases'][:5]))
            lines.append("")
        
        # Source Link
        if sop_link:
            lines.append(f"🔗 **Source:** [View Document]({sop_link})")
            lines.append("")

        # Confidence
        if summary.get('confidence'):
            confidence_pct = int(summary['confidence'] * 100)
            lines.append(f"*Relevance: {confidence_pct}%*")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_instruction_response(sop_title: str, instructions: List[Dict], sop_link: str = None) -> str:
        """Format instruction response with proper numbering."""
        if not instructions:
            return f"**{sop_title}**\n\nNo step-by-step instructions found in this SOP."
        
        lines = []
        lines.append(f"📋 **Step-by-Step Guide: {sop_title}**")
        lines.append("")
        
        step_counter = 1
        for inst in instructions:
            if inst['type'] == 'numbered':
                lines.append(f"{inst.get('step', step_counter)}. {inst['text']}")
                step_counter = inst.get('step', step_counter) + 1
            else:
                lines.append(f"• {inst['text']}")
        
        if sop_link:
            lines.append("")
            lines.append(f"🔗 **Source:** [View Document]({sop_link})")
            
        return "\n".join(lines)
    
    @staticmethod
    def format_answer_response(answer: Dict[str, Any]) -> str:
        """Format question answer response."""
        lines = []
        
        confidence = answer.get('confidence', 0)
        
        if confidence > 0.7:
            lines.append("**Answer:**")
        elif confidence > 0.4:
            lines.append("**I think:**")
        else:
            lines.append("**Based on available information:**")
        
        lines.append(answer.get('answer', ''))
        
        if answer.get('source'):
            lines.append(f"*(Source: {answer['source']})*")
        
        if answer.get('suggestions'):
            lines.append("")
            lines.append("**You might also ask:**")
            for suggestion in answer['suggestions']:
                lines.append(f"• {suggestion}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_greeting_response(user_id: str = None) -> str:
        """Format greeting response."""
        name = user_id or "there"
        return f"Hello {name}! I'm your SOP assistant. I can help you find and understand security procedures. What are you looking for today?"
    
    @staticmethod
    def format_thanks_response() -> str:
        """Format thank you response."""
        return "You're welcome! Let me know if you need help with anything else."
    
    @staticmethod
    def format_error_response(error: str) -> str:
        """Format error response."""
        return f"I encountered an issue: {error}. Please try rephrasing your question."


##############################################################################
# LANGGRAPH CONVERSATION STATE
##############################################################################

class ConversationState(TypedDict):
    """LangGraph conversation state."""
    conversation_id: str
    user_id: Optional[str]
    
    # Current query
    query: str
    
    # DeriveEngine results
    derive_results: Optional[Dict[str, Any]]
    active_sop: Optional[Dict[str, Any]]
    sop_history: List[Dict[str, Any]]
    
    # Processed data
    summary: Optional[Dict[str, Any]]
    instructions: List[Dict[str, Any]]
    answer: Optional[Dict[str, Any]]
    
    # Conversation state
    messages: List[Dict[str, Any]]
    intent: Optional[str]
    awaiting_counter_response: bool
    pending_counter_question: Optional[str]
    
    # Response
    response: str
    error: Optional[str]


##############################################################################
# ENTERPRISE SOP ASSISTANT - MAIN CLASS
##############################################################################

class EnterpriseSOPAssistant:
    """
    Enterprise SOP Assistant - Lightweight Hybrid Architecture.
    
    Architecture:
    1. User Query → DeriveEngine (retrieval)
    2. Hybrid Summarizer (SentenceTransformer + NLP)
    3. Instruction Extractor (rule-based)
    4. Counter-Question Logic
    5. Response Formatter
    6. LangGraph Orchestration
    """
    
    def __init__(self, 
                 derive_engine: DeriveEngine, 
                 config: SOPAssistantConfig = None,
                 transformer_model: Any = None):
        """Initialize Assistant with DeriveEngine and Configuration."""
        self.derive_engine = derive_engine
        self.config = config or SOPAssistantConfig()
        
        # Core components
        self.classifier = IntentClassifier()
        self.summarizer = SOPExtractiveSummarizer(
            self.config, 
            transformer_model=transformer_model
        )
        self.formatter = ResponseFormatter()
        
        # LangGraph
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.conversations = {}
        
        logger.info("Enterprise SOP Assistant initialized")
        logger.info(f"Config: SentenceTransformer={self.config.sentence_transformer_model}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("retrieve_sops", self._retrieve_sops)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("extract_instructions", self._extract_instructions)
        workflow.add_node("answer_question", self._answer_question)
        workflow.add_node("handle_counter_response", self._handle_counter_response)
        workflow.add_node("format_response", self._format_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "new_sop_query": "retrieve_sops",
                "sop_summary": "generate_summary",
                "instruction": "extract_instructions",
                "clarification": "answer_question",
                "counter_question": "handle_counter_response",
                "greeting": "format_response",
                "thanks": "format_response",
                "comparison": "retrieve_sops",
                "unknown": "retrieve_sops"
            }
        )
        
        # Edges from retrieval
        workflow.add_edge("retrieve_sops", "generate_summary")
        workflow.add_edge("generate_summary", "format_response")
        workflow.add_edge("extract_instructions", "format_response")
        workflow.add_edge("answer_question", "format_response")
        workflow.add_edge("handle_counter_response", "retrieve_sops")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _classify_intent(self, state: ConversationState) -> ConversationState:
        """Classify user intent."""
        query = state["query"]
        has_active_sop = state.get("active_sop") is not None
        awaiting_response = state.get("awaiting_counter_response", False)
        
        intent = self.classifier.classify(query, has_active_sop, awaiting_response)
        state["intent"] = intent.value
        
        logger.info(f"Intent: {intent.value} | Query: {query[:50]}...")
        return state
    
    def _route_by_intent(self, state: ConversationState) -> str:
        """Route to appropriate node based on intent."""
        return state.get("intent", "unknown")
    
    def _retrieve_sops(self, state: ConversationState) -> ConversationState:
        """Retrieve SOPs using DeriveEngine."""
        query = state["query"]
        
        try:
            # Call your existing DeriveEngine
            derive_results = self.derive_engine.derive(
                query=query,
                top_k=self.config.derive_top_k,
                threat_type=self.config.derive_threat_type,
                category=self.config.derive_category
            )
            
            state["derive_results"] = derive_results
            
            # Set active SOP
            if derive_results.get("results"):
                top_result = derive_results["results"][0]
                state["active_sop"] = top_result
                
                # Update SOP history
                if "sop_history" not in state:
                    state["sop_history"] = []
                
                sop_id = top_result.get("sop_id")
                existing_ids = [s.get("sop_id") for s in state["sop_history"]]
                
                if sop_id not in existing_ids:
                    state["sop_history"].append(top_result)
                    logger.info(f"Active SOP set: {top_result.get('title')}")
            
            # Reset counter-question flag
            state["awaiting_counter_response"] = False
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["error"] = str(e)
        
        return state
    
    def _generate_summary(self, state: ConversationState) -> ConversationState:
        """Generate extractive summary using SentenceTransformer + NLP."""
        active_sop = state.get("active_sop")
        
        if not active_sop:
            state["response"] = "No active SOP to summarize. Please search for one first."
            return state
        
        # Get chunks from active SOP
        chunks = active_sop.get("matched_chunks", [])
        
        if not chunks:
            state["response"] = "No content available for this SOP."
            return state
        
        # Determine summary type based on intent
        intent = state.get("intent", "")
        if "clarification" in intent:
            summary_type = "query-focused"
        elif "summary" in intent and "quick" in state.get("query", "").lower():
            summary_type = "quick"
        else:
            summary_type = "general"
        
        # Generate summary
        summary = self.summarizer.summarize(
            sop_chunks=chunks,
            query=state.get("query") if summary_type == "query-focused" else None,
            summary_type=summary_type
        )
        
        state["summary"] = summary
        
        # Generate counter-question if enabled
        if self.config.enable_counter_questions and summary_type != "quick":
            state["pending_counter_question"] = "Would you like to see the step-by-step instructions?"
            state["awaiting_counter_response"] = True
        
        return state
    
    def _extract_instructions(self, state: ConversationState) -> ConversationState:
        """Extract step-by-step instructions using NLP rules."""
        active_sop = state.get("active_sop")
        
        if not active_sop:
            return self._retrieve_sops(state)
        
        chunks = active_sop.get("matched_chunks", [])
        instructions = self.summarizer.extract_instructions(chunks)
        
        state["instructions"] = instructions
        state["awaiting_counter_response"] = False
        
        return state
    
    def _answer_question(self, state: ConversationState) -> ConversationState:
        """Answer specific questions using SentenceTransformer similarity."""
        query = state["query"]
        active_sop = state.get("active_sop")
        
        if not active_sop:
            return self._retrieve_sops(state)
        
        chunks = active_sop.get("matched_chunks", [])
        
        if not chunks:
            state["answer"] = {
                "answer": "No information available to answer your question.",
                "confidence": 0.0
            }
            return state
        
        # Answer question
        answer = self.summarizer.answer_question(query, chunks)
        state["answer"] = answer
        state["awaiting_counter_response"] = False
        
        return state
    
    def _handle_counter_response(self, state: ConversationState) -> ConversationState:
        """Handle positive response to counter-question."""
        query = state["query"].lower()
        
        # User agreed to see instructions
        if any(word in query for word in ['yes', 'yeah', 'sure', 'ok', 'please']):
            state["intent"] = "instruction"
            return self._extract_instructions(state)
        
        # User declined - just return to current SOP
        state["awaiting_counter_response"] = False
        state["response"] = "Okay, let me know if you need anything else about this SOP."
        
        return state
    
    def _format_response(self, state: ConversationState) -> ConversationState:
        """Format final response using NLP formatter."""
        intent = state.get("intent", "")
        active_sop = state.get("active_sop")
        
        try:
            if intent == "greeting":
                state["response"] = self.formatter.format_greeting_response(state.get("user_id"))
            
            elif intent == "thanks":
                state["response"] = self.formatter.format_thanks_response()
            
            elif intent in ["sop_summary", "new_sop_query"] and state.get("summary"):
                sop_title = active_sop.get("title", "SOP") if active_sop else "SOP"
                sop_link = active_sop.get("sop_link") if active_sop else None
                state["response"] = self.formatter.format_summary_response(
                    sop_title, state["summary"], sop_link
                )
                
                # Append counter-question if pending
                if state.get("pending_counter_question") and state.get("awaiting_counter_response"):
                    state["response"] += f"\n\n{state['pending_counter_question']}"
            
            elif intent == "instruction" and state.get("instructions"):
                sop_title = active_sop.get("title", "SOP") if active_sop else "SOP"
                sop_link = active_sop.get("sop_link") if active_sop else None
                state["response"] = self.formatter.format_instruction_response(
                    sop_title, state["instructions"], sop_link
                )
            
            elif intent in ["clarification", "question"] and state.get("answer"):
                state["response"] = self.formatter.format_answer_response(state["answer"])
                
                # Append link to clarifications too if available
                sop_link = active_sop.get("sop_link") if active_sop else None
                if sop_link:
                    state["response"] += f"\n\n🔗 **Source:** [View Document]({sop_link})"
            
            elif state.get("derive_results"):
                results = state["derive_results"].get("results", [])
                if results:
                    titles = [r.get("title", "Unknown") for r in results[:3]]
                    state["response"] = f"I found {len(results)} relevant SOPs:\n"
                    for i, title in enumerate(titles, 1):
                        state["response"] += f"{i}. {title}\n"
                    state["response"] += "\nWhich one would you like to know more about?"
                else:
                    state["response"] = "I couldn't find any relevant SOPs. Please try a different search."
            
            elif state.get("error"):
                state["response"] = self.formatter.format_error_response(state["error"])
            
            else:
                state["response"] = "How can I help you with SOPs today?"
        
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            state["response"] = "I encountered an issue processing your request. Please try again."
        
        # Update conversation history
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append({
            "role": "user",
            "content": state["query"],
            "timestamp": time.time()
        })
        
        state["messages"].append({
            "role": "assistant",
            "content": state["response"],
            "intent": state.get("intent"),
            "timestamp": time.time()
        })
        
        # Limit history
        if len(state["messages"]) > self.config.max_conversation_turns * 2:
            state["messages"] = state["messages"][-self.config.max_conversation_turns * 2:]
        
        return state
    
    def chat(self, 
             query: str, 
             conversation_id: str = None,
             user_id: str = None) -> Dict[str, Any]:
        """
        Main chat method - complete conversation flow.
        
        Args:
            query: User's message
            conversation_id: Existing conversation ID (for follow-ups)
            user_id: Optional user identifier
            
        Returns:
            Response dictionary with answer and context
        """
        # Generate or reuse conversation ID
        if not conversation_id:
            conversation_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()
        
        # Get or create state
        if conversation_id in self.conversations:
            state = self.conversations[conversation_id].copy()
        else:
            state = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "query": query,
                "derive_results": None,
                "active_sop": None,
                "sop_history": [],
                "summary": None,
                "instructions": [],
                "answer": None,
                "messages": [],
                "intent": None,
                "awaiting_counter_response": False,
                "pending_counter_question": None,
                "response": "",
                "error": None
            }
        
        # Update with new query
        state["query"] = query
        
        # Run through graph
        try:
            config = {"configurable": {"thread_id": conversation_id}}
            result = self.graph.invoke(state, config)
            
            # Store updated state
            self.conversations[conversation_id] = result
            
            return {
                "response": result.get("response", ""),
                "conversation_id": conversation_id,
                "active_sop": result.get("active_sop", {}).get("title") if result.get("active_sop") else None,
                "intent": result.get("intent"),
                "has_error": result.get("error") is not None,
                "sop_count": len(result.get("derive_results", {}).get("results", [])) if result.get("derive_results") else 0
            }
            
        except Exception as e:
            logger.exception(f"Chat failed: {e}")
            return {
                "response": f"I encountered an error: {str(e)[:100]}. Please try again.",
                "conversation_id": conversation_id,
                "active_sop": None,
                "intent": None,
                "has_error": True,
                "sop_count": 0
            }
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history."""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id].get("messages", [])
        return []


def create_enterprise_assistant(derive_engine: DeriveEngine, 
                               config: SOPAssistantConfig = None,
                               transformer_model: Any = None,
                               **config_overrides) -> EnterpriseSOPAssistant:
    """Factory function for EnterpriseSOPAssistant."""
    config = config or SOPAssistantConfig()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key: {key}")
    
    return EnterpriseSOPAssistant(
        derive_engine=derive_engine, 
        config=config,
        transformer_model=transformer_model
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Enterprise SOP Assistant - Lightweight Hybrid Architecture")
    print("=" * 60)
    print("\nComponents:")
    print("  • DeriveEngine: Your existing retrieval system")
    print("  • SentenceTransformer: all-MiniLM-L6-v2 (sentence ranking only)")
    print("  • NLTK + spaCy: Rule-based NLP extraction")
    print("  • LangGraph: Conversation orchestration")
    print("\nNo duplicate retrieval, no generative models, no external APIs")
    print("=" * 60)
