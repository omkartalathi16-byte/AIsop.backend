"""
NN Engine — Part 2: Summarization & Question Answering
Uses local HuggingFace transformer models (no LLM API).
"""
import re
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)


class NNEngine:
    """Summarizes SOP content, extracts steps, and answers follow-up questions."""

    def __init__(self):
        logger.info("Loading summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU; set to 0 for GPU
        )

        logger.info("Loading QA model...")
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1
        )
        logger.info("NN Engine models loaded.")

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Summarize SOP content into a concise paragraph."""
        if len(text.split()) < min_length:
            return text

        try:
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:500]

    def extract_steps(self, text: str) -> list[str]:
        """Extract numbered action steps from SOP content."""
        # Try to find existing numbered steps
        numbered = re.findall(r'(?:^|\n)\s*(\d+[\.\)]\s*.+?)(?=\n\s*\d+[\.\)]|\n\n|$)', text, re.DOTALL)
        if numbered:
            return [step.strip() for step in numbered]

        # Try bullet points
        bullets = re.findall(r'(?:^|\n)\s*[-•*]\s*(.+)', text)
        if bullets:
            return [b.strip() for b in bullets]

        # Fall back: split sentences and label as steps
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        return [f"Step {i+1}: {s}." for i, s in enumerate(sentences[:8])]

    def answer_question(self, question: str, context: str) -> str:
        """Answer a follow-up question using SOP content as context."""
        try:
            result = self.qa_model(
                question=question,
                context=context,
                max_answer_len=200
            )
            if result["score"] < 0.05:
                return "I couldn't find a specific answer in the SOP. Could you rephrase your question?"
            return result["answer"]
        except Exception as e:
            logger.error(f"QA failed: {e}")
            return "Sorry, I was unable to process your question against the SOP context."
