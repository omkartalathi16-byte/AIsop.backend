"""
Enterprise-grade LLM Service for RAG Applications - Optimized for Intel i5-4200U
Version: 2.0.1 (Hardware Optimized)
Author: Enterprise AI Team
License: Proprietary

Hardware Profile:
- CPU: Intel i5-4200U (2 cores, 4 threads)
- Memory: 8GB RAM
- Optimization: CPU_OPTIMIZED with manual thread control
"""

import logging
import os
import json
import hashlib
import pickle
import time
import threading
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from functools import lru_cache, wraps
from contextlib import contextmanager
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import re

# Third-party imports for enterprise features
import jinja2
import psutil
import redis
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
import numpy as np
from cachetools import TTLCache, LRUCache
import xxhash

# Local imports
from llama_cpp import Llama, LlamaGrammar, LogitsProcessorList

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
LLM_REQUESTS = Counter('llm_requests_total', 'Total LLM requests', ['model', 'mode'])
LLM_TOKENS = Counter('llm_tokens_total', 'Total tokens processed', ['type'])
LLM_LATENCY = Histogram('llm_latency_seconds', 'LLM request latency', ['operation'])
LLM_ERRORS = Counter('llm_errors_total', 'Total LLM errors', ['type'])
LLM_CACHE_HITS = Counter('llm_cache_hits_total', 'Cache hits')
LLM_CONTEXT_SIZE = Gauge('llm_context_size', 'Current context window usage')
LLM_MEMORY_USAGE = Gauge('llm_memory_bytes', 'Memory usage in bytes')

class ResponseMode(str, Enum):
    """Enterprise response modes with specific use cases"""
    CONCISE = "concise"
    DETAILED = "detailed"
    STRICT = "strict"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CODE = "code"
    SUMMARIZATION = "summarization"

class ModelProfile(str, Enum):
    """Hardware-optimized model profiles"""
    CPU_OPTIMIZED = "cpu_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    BALANCED = "balanced"
    HIGH_THROUGHPUT = "high_throughput"

class AuditLevel(str, Enum):
    """Audit logging levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"

@dataclass
class GenerationConfig:
    """Advanced generation configuration with validation"""
    max_tokens: int = 256
    temperature: float = 0.2  # Lowered for enterprise
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    logit_bias: Dict[int, float] = field(default_factory=dict)
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    typical_p: float = 1.0
    tfs_z: float = 1.0
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @classmethod
    def for_mode(cls, mode: ResponseMode) -> 'GenerationConfig':
        """Get enterprise-optimized configuration for specific response mode"""
        configs = {
            ResponseMode.CONCISE: cls(
                max_tokens=128,
                temperature=0.1,
                top_p=0.85,
                repeat_penalty=1.2,
                stop_sequences=["\n\n", "###", "Question:"]
            ),
            ResponseMode.DETAILED: cls(
                max_tokens=512,
                temperature=0.3,
                top_p=0.92,
                repeat_penalty=1.1,
                frequency_penalty=0.1
            ),
            ResponseMode.STRICT: cls(
                max_tokens=256,
                temperature=0.01,
                top_p=0.95,
                repeat_penalty=1.3,
                presence_penalty=-0.1
            ),
            ResponseMode.ANALYTICAL: cls(
                max_tokens=768,
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.15,
                frequency_penalty=0.05
            ),
            ResponseMode.CREATIVE: cls(
                max_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                repeat_penalty=1.0,
                frequency_penalty=0.3
            ),
            ResponseMode.CODE: cls(
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repeat_penalty=1.0,
                stop_sequences=["```", "\n\n\n"]
            ),
            ResponseMode.SUMMARIZATION: cls(
                max_tokens=256,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.2,
                frequency_penalty=0.1
            )
        }
        return configs.get(mode, cls())

class ContextWindowManager:
    """Sophisticated context window management with dynamic optimization"""
    
    def __init__(
        self,
        max_context_tokens: int = 1024,  # Optimized for i5-4200U
        reserved_tokens: int = 150,      # For response
        system_prompt_tokens: int = 50,
        safety_margin: float = 0.1       # 10% safety margin
    ):
        self.max_context_tokens = max_context_tokens
        self.reserved_tokens = reserved_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.safety_margin = safety_margin
        self.available_tokens = self._calculate_available_tokens()
        
    def _calculate_available_tokens(self) -> int:
        """Calculate available tokens for context"""
        max_usable = self.max_context_tokens - self.reserved_tokens
        return int(max_usable * (1 - self.safety_margin))
    
    def optimize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        token_counter: Callable[[str], int]
    ) -> List[Dict[str, Any]]:
        """
        Dynamically optimize chunks to fit context window
        Using advanced selection algorithm
        """
        if not chunks:
            return []
        
        # Score chunks by relevance and priority
        scored_chunks = []
        for chunk in chunks:
            score = self._calculate_chunk_score(chunk, query)
            token_count = token_counter(chunk.get('content', ''))
            scored_chunks.append((score, token_count, chunk))
        
        # Sort by score descending
        scored_chunks.sort(reverse=True)
        
        # Select chunks greedily with token budget
        selected_chunks = []
        total_tokens = 0
        token_budget = self.available_tokens - token_counter(query)
        
        for score, token_count, chunk in scored_chunks:
            if total_tokens + token_count <= token_budget:
                selected_chunks.append(chunk)
                total_tokens += token_count
            else:
                # Try to truncate if it's the highest scored chunk
                if not selected_chunks and token_count > token_budget:
                    truncated = self._truncate_chunk(chunk, token_budget, token_counter)
                    if truncated:
                        selected_chunks.append(truncated)
                break
        
        LLM_CONTEXT_SIZE.set(total_tokens)
        return selected_chunks
    
    def _calculate_chunk_score(self, chunk: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for chunk"""
        metadata = chunk.get('metadata', {})
        
        # Base relevance score
        base_score = metadata.get('relevance_score', 0.5)
        
        # Boost for recency
        timestamp = metadata.get('timestamp')
        if timestamp:
            recency_boost = self._calculate_recency_boost(timestamp)
            base_score *= (1 + recency_boost)
        
        # Boost for source authority
        source_authority = metadata.get('source_authority', 1.0)
        base_score *= source_authority
        
        return base_score
    
    def _calculate_recency_boost(self, timestamp: datetime) -> float:
        """Calculate recency boost factor"""
        age = datetime.now() - timestamp
        days = age.days
        if days < 7:
            return 0.3
        elif days < 30:
            return 0.15
        elif days < 90:
            return 0.05
        return 0.0
    
    def _truncate_chunk(
        self,
        chunk: Dict[str, Any],
        target_tokens: int,
        token_counter: Callable[[str], int]
    ) -> Optional[Dict[str, Any]]:
        """Intelligently truncate chunk to fit token budget"""
        content = chunk.get('content', '')
        
        # Semantic truncation - keep first and last parts
        words = content.split()
        if len(words) < 50:
            return None
        
        # Keep first 30% and last 20% for most context
        first_part = words[:int(len(words) * 0.3)]
        last_part = words[-int(len(words) * 0.2):]
        
        truncated_content = ' '.join(first_part + ['...'] + last_part)
        
        # Check if it fits now
        if token_counter(truncated_content) <= target_tokens:
            chunk['content'] = truncated_content
            chunk['truncated'] = True
            return chunk
        
        return None

class PromptTemplateManager:
    """Advanced prompt template management with Jinja2"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path("templates")
        self.template_dir.mkdir(exist_ok=True)
        
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.env.filters['truncate'] = self.truncate_filter
        self.env.filters['format_json'] = self.format_json_filter
        self.env.filters['highlight'] = self.highlight_filter
        
        # Initialize default templates
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default prompt templates"""
        templates = {
            'rag_base.j2': """
<|im_start|>system
{{ system_prompt }}

{% if guidelines %}
Guidelines:
{% for guideline in guidelines %}
- {{ guideline }}
{% endfor %}
{% endif %}

Context Information:
{% for chunk in context_chunks %}
{% if chunk.metadata.title %}
Source {{ loop.index }}: {{ chunk.metadata.title }}
{% endif %}
{{ chunk.content|truncate(500) }}
{% endfor %}
<|im_end|>

{% for message in history %}
<|im_start|>{{ message.role }}
{{ message.content }}
<|im_end|>
{% endfor %}

<|im_start|>user
{{ query }}
<|im_end|>

<|im_start|>assistant
""",
            
            'rag_structured.j2': """
<|im_start|>system
{{ system_prompt }}

You must respond in the following JSON format:
{
    "answer": "your answer here",
    "sources": ["source1", "source2"],
    "confidence": 0.0-1.0,
    "missing_information": ["what's not in context"]
}
<|im_end|>

<|im_start|>user
Query: {{ query }}

Context:
{% for chunk in context_chunks %}
[{{ loop.index }}] {{ chunk.content }}
{% endfor %}
<|im_end|>

<|im_start|>assistant
"""
        }
        
        for name, content in templates.items():
            template_path = self.template_dir / name
            if not template_path.exists():
                template_path.write_text(content.strip())
    
    def render(
        self,
        template_name: str,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        guidelines: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Render template with context"""
        template = self.env.get_template(template_name)
        
        return template.render(
            query=query,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
            history=history or [],
            guidelines=guidelines or [],
            **kwargs
        )
    
    @staticmethod
    def truncate_filter(text: str, length: int = 500) -> str:
        """Truncate text to specified length"""
        if len(text) <= length:
            return text
        return text[:length] + "..."
    
    @staticmethod
    def format_json_filter(data: Any) -> str:
        """Format data as JSON"""
        return json.dumps(data, indent=2)
    
    @staticmethod
    def highlight_filter(text: str, term: str) -> str:
        """Highlight search terms (basic implementation)"""
        if not term:
            return text
        return re.sub(f'({term})', r'**\1**', text, flags=re.IGNORECASE)

class CacheManager:
    """Multi-layer caching with Redis fallback"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        memory_cache_size: int = 1000,
        memory_ttl: int = 3600,
        redis_ttl: int = 86400
    ):
        self.memory_cache = TTLCache(maxsize=memory_cache_size, ttl=memory_ttl)
        self.redis_client = None
        self.redis_ttl = redis_ttl
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback"""
        # Try memory cache first
        if key in self.memory_cache:
            LLM_CACHE_HITS.inc()
            return self.memory_cache[key]
        
        # Try Redis
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    LLM_CACHE_HITS.inc()
                    result = pickle.loads(data.encode('latin1'))
                    # Update memory cache
                    self.memory_cache[key] = result
                    return result
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Set in both caches"""
        # Memory cache
        self.memory_cache[key] = value
        
        # Redis
        if self.redis_client:
            try:
                data = pickle.dumps(value).decode('latin1')
                self.redis_client.setex(key, self.redis_ttl, data)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        hash_input = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return xxhash.xxh64(hash_input.encode()).hexdigest()

class CircuitBreaker:
    """Circuit breaker pattern for resilience"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time >= self.recovery_timeout:
                        self.state = "HALF_OPEN"
                        logger.info("Circuit breaker half-open")
                    else:
                        raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                with self.lock:
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                        logger.info("Circuit breaker closed")
                
                return result
                
            except self.expected_exception as e:
                with self.lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        logger.warning(f"Circuit breaker opened: {e}")
                
                raise
        
        return wrapper

class AuditLogger:
    """Audit logging for compliance"""
    
    def __init__(self, log_dir: Path, level: AuditLevel = AuditLevel.BASIC):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = level
        self.session_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        
        # Setup audit log file
        self.audit_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.log"
    
    def log_request(
        self,
        user_id: str,
        query: str,
        model: str,
        context_chunks: int,
        response_time: float,
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """Log request for audit purposes"""
        if self.level == AuditLevel.BASIC and not success:
            return  # Only log failures in basic mode
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'user_id': user_id,
            'event_type': 'llm_request',
            'data': {
                'model': model,
                'query_length': len(query),
                'context_chunks': context_chunks,
                'response_time': response_time,
                'success': success
            }
        }
        
        # Add more details for higher audit levels
        if self.level == AuditLevel.DETAILED and metadata:
            audit_entry['data']['metadata'] = metadata
        
        if self.level == AuditLevel.FULL:
            audit_entry['data']['query_preview'] = query[:200]
        
        # Write to audit log
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        # Also log to security system if needed
        if not success and self.level != AuditLevel.BASIC:
            self._alert_security_team(audit_entry)
    
    def _alert_security_team(self, entry: Dict):
        """Alert security team about suspicious activity"""
        # Implementation would integrate with SIEM, Slack, email, etc.
        logger.warning(f"Security alert: {entry}")

class EnterpriseLLMService:
    """
    Enterprise-grade LLM Service with advanced features
    Optimized for Intel i5-4200U CPU
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        model_profile: ModelProfile = ModelProfile.CPU_OPTIMIZED,
        redis_url: Optional[str] = None,
        audit_level: AuditLevel = AuditLevel.BASIC,
        enable_monitoring: bool = False,  # DISABLED by default for i5-4200U
        enable_caching: bool = True,
        enable_circuit_breaker: bool = True
    ):
        # Core configuration
        self.model_path = str(model_path)
        self.model_profile = model_profile
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Initialize components
        self.llm: Optional[Llama] = None
        self.tokenizer = None
        self.context_manager = ContextWindowManager(max_context_tokens=self._get_optimal_context_size())
        self.template_manager = PromptTemplateManager()
        self.cache_manager = CacheManager(redis_url=redis_url) if enable_caching else None
        self.audit_logger = AuditLogger(Path("logs/audit"), audit_level)
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Metrics and monitoring
        self.metrics = defaultdict(float)
        self.metrics_lock = threading.Lock()
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'errors': 0,
            'cache_hits': 0
        }
        
        # Health check
        self.health_status = {
            'status': 'initializing',
            'last_check': None,
            'error_count': 0,
            'avg_latency': 0.0
        }
        
        # Start monitoring thread if enabled
        if enable_monitoring:
            self._start_monitoring()
        
        # Initialize model
        self._initialize_model()
    
    def _get_optimal_context_size(self) -> int:
        """Get optimal context size based on hardware profile"""
        profiles = {
            ModelProfile.CPU_OPTIMIZED: 1024,  # Increased to 1024 for better RAG
            ModelProfile.MEMORY_OPTIMIZED: 512,
            ModelProfile.BALANCED: 1024,
            ModelProfile.HIGH_THROUGHPUT: 512
        }
        return profiles.get(self.model_profile, 1024)
    
    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on hardware profile"""
        cpu_count = psutil.cpu_count(logical=False)
        total_cpus = psutil.cpu_count(logical=True)
        
        profiles = {
            ModelProfile.CPU_OPTIMIZED: 2,  # HARDCODED to 2 for i5-4200U
            ModelProfile.MEMORY_OPTIMIZED: max(1, cpu_count // 2),
            ModelProfile.BALANCED: cpu_count,
            ModelProfile.HIGH_THROUGHPUT: total_cpus
        }
        return profiles.get(self.model_profile, 2)
    
    def _initialize_model(self):
        """Initialize Llama model with enterprise optimizations"""
        try:
            # Setup Windows dependencies if needed
            self._setup_windows_dependencies()
            
            # Check model exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            logger.info(f"Initializing enterprise LLM from {self.model_path}")
            
            # Hardware-aware initialization
            n_threads = self._get_optimal_threads()
            n_ctx = self._get_optimal_context_size()
            
            logger.info(f"Profile: {self.model_profile.value}, Threads: {n_threads}, Context: {n_ctx}")
            
            # Track memory usage
            process = psutil.Process()
            mem_before = process.memory_info().rss
            
            # Initialize model with enterprise settings
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=256,  # REDUCED from 512 to 256 for better memory usage
                n_gpu_layers=0,  # CPU-only for stability
                use_mmap=True,
                use_mlock=False,  # Avoid memory locking issues
                verbose=False,
                seed=42,
                f16_kv=True,  # Use half-precision for key/value cache
                logits_all=False,  # Save memory
                embedding=False,  # We're not generating embeddings
                last_n_tokens_size=64,  # For repetition penalty
                rope_scaling_type=None,
            )
            
            # Track memory after
            mem_after = process.memory_info().rss
            mem_used = mem_after - mem_before
            LLM_MEMORY_USAGE.set(mem_used)
            
            self.health_status['status'] = 'healthy'
            logger.info(f"Model initialized successfully (memory: {mem_used / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            self.health_status['status'] = 'unhealthy'
            self.health_status['error'] = str(e)
            logger.error(f"Model initialization failed: {e}", exc_info=True)
            raise
    
    def _setup_windows_dependencies(self):
        """Setup Windows dependencies with multiple fallbacks"""
        if os.name != 'nt':
            return
        
        # Multiple possible MinGW locations
        mingw_paths = [
            "C:\\mingw\\bin",
            "C:\\mingw64\\bin",
            "C:\\msys64\\mingw64\\bin",
            "C:\\msys64\\ucrt64\\bin",
            os.path.expanduser("~\\scoop\\apps\\mingw\\current\\bin")
        ]
        
        for mingw_bin in mingw_paths:
            if os.path.exists(mingw_bin):
                current_path = os.environ.get("PATH", "")
                if mingw_bin not in current_path:
                    os.environ["PATH"] = mingw_bin + os.pathsep + current_path
                    logger.info(f"Added {mingw_bin} to PATH")
                break
    
    def _token_counter(self, text: str) -> int:
        """Estimate token count (simplified)"""
        if not text:
            return 0
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4 + 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _generate_with_retry(self, *args, **kwargs):
        """Generate with retry logic"""
        if self.circuit_breaker:
            return self.circuit_breaker(self.llm)(*args, **kwargs)
        return self.llm(*args, **kwargs)
    
    # Removed @LLM_LATENCY.time() because it fails on metrics with labels
    def generate_response(
        self,
        query: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        config: Optional[GenerationConfig] = None,
        mode: ResponseMode = ResponseMode.CONCISE,
        template_name: str = "rag_base.j2",
        user_id: str = "anonymous",
        metadata: Optional[Dict] = None,
        stream: bool = False,
        grammar: Optional[LlamaGrammar] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], Any]:
        """
        Generate response with enterprise-grade features
        
        Args:
            query: User query
            system_prompt: System instructions
            history: Conversation history
            config: Generation configuration
            mode: Response mode
            template_name: Template to use
            user_id: User identifier for audit
            metadata: Additional metadata
            stream: Enable streaming
            grammar: Grammar for constrained generation
        
        Returns:
            Generated response or stream iterator
        """
        start_time = time.time()
        LLM_REQUESTS.labels(model=self.model_profile.value, mode=mode.value).inc()
        
        # Update stats
        with self.metrics_lock:
            self.stats['total_requests'] += 1
        
        # Use mode-specific config if not provided
        if not config:
            config = GenerationConfig.for_mode(mode)
        
        # Override config with any provided kwargs (e.g., max_tokens, temperature)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Check cache first
        if self.enable_caching and self.cache_manager and not stream:
            cache_key = self.cache_manager.generate_key(
                "llm_response",
                query,
                system_prompt,
                history,
                config.max_tokens,
                mode.value,
                user_id
            )
            
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.stats['cache_hits'] += 1
                LLM_CACHE_HITS.inc()
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_response
        
        try:
            if not self.llm:
                raise RuntimeError("LLM service not initialized")
            
            # Format prompt using template
            formatted_prompt = self.template_manager.render(
                template_name,
                query=query,
                context_chunks=[],  # No context for basic generation
                system_prompt=system_prompt,
                history=history
            )
            
            # Prepare generation parameters
            gen_kwargs = {
                "prompt": formatted_prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stop": config.stop_sequences or ["<|im_end|>", "<|im_start|>"],
                "echo": False,
                "stream": stream,
                "logit_bias": config.logit_bias,
                "mirostat_mode": config.mirostat_mode,
                "mirostat_tau": config.mirostat_tau,
                "mirostat_eta": config.mirostat_eta,
                "typical_p": config.typical_p,
                "tfs_z": config.tfs_z
            }
            
            if grammar:
                gen_kwargs["grammar"] = grammar
            
            # Generate with retry
            response = self._generate_with_retry(**gen_kwargs)
            
            # Update token stats
            if not stream and 'usage' in response:
                tokens_used = response['usage'].get('total_tokens', 0)
                self.stats['total_tokens'] += tokens_used
                LLM_TOKENS.labels(type='total').inc(tokens_used)
            
            # Process response
            if stream:
                return self._handle_stream_response(response, cache_key if self.enable_caching else None)
            
            result = response['choices'][0]['text'].strip()
            
            # Cache successful response
            if self.enable_caching and self.cache_manager and result:
                self.cache_manager.set(cache_key, result)
            
            # Audit logging
            elapsed = time.time() - start_time
            self.audit_logger.log_request(
                user_id=user_id,
                query=query,
                model=self.model_profile.value,
                context_chunks=0,
                response_time=elapsed,
                success=True,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            LLM_ERRORS.labels(type=type(e).__name__).inc()
            
            with self.metrics_lock:
                self.stats['errors'] += 1
                self.health_status['error_count'] += 1
            
            # Audit failure
            elapsed = time.time() - start_time
            self.audit_logger.log_request(
                user_id=user_id,
                query=query,
                model=self.model_profile.value,
                context_chunks=0,
                response_time=elapsed,
                success=False,
                metadata={'error': str(e)}
            )
            
            logger.error(f"Generation failed: {e}", exc_info=True)
            
            # Graceful degradation
            return self._get_fallback_response(query, e)
    
    def synthesize_rag_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        mode: ResponseMode = ResponseMode.CONCISE,
        template_name: str = "rag_base.j2",
        user_id: str = "anonymous",
        guidelines: Optional[List[str]] = None,
        include_metadata: bool = False,
        structured_output: bool = False,
        metadata: Optional[Dict] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Synthesize RAG response with advanced context optimization
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            mode: Response mode
            template_name: Template to use
            user_id: User identifier
            guidelines: Response guidelines
            include_metadata: Include metadata in response
            structured_output: Return structured response
            metadata: Additional metadata
        """
        start_time = time.time()
        LLM_REQUESTS.labels(model=self.model_profile.value, mode=mode.value).inc()
        
        with self.metrics_lock:
            self.stats['total_requests'] += 1
        
        # Generate cache key
        if self.enable_caching and self.cache_manager:
            cache_key = self.cache_manager.generate_key(
                "rag_response",
                query,
                [c.get('id', '') for c in context_chunks[:10]],
                mode.value,
                structured_output,
                user_id
            )
            
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.stats['cache_hits'] += 1
                LLM_CACHE_HITS.inc()
                return cached_response
        
        try:
            if not self.llm:
                raise RuntimeError("LLM service not initialized")
            
            # Optimize context chunks
            optimized_chunks = self.context_manager.optimize_chunks(
                context_chunks,
                query,
                self._token_counter
            )
            
            if not optimized_chunks:
                logger.warning("No context chunks after optimization")
            
            # Select appropriate template
            if structured_output:
                template_name = "rag_structured.j2"
            
            # Render prompt
            formatted_prompt = self.template_manager.render(
                template_name,
                query=query,
                context_chunks=optimized_chunks,
                system_prompt=self._get_system_prompt(mode),
                guidelines=guidelines or [],
                history=history or []
            )
            
            # Get config for mode
            config = GenerationConfig.for_mode(mode)
            
            # Generate response
            response = self._generate_with_retry(
                formatted_prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop_sequences or ["<|im_end|>", "<|im_start|>"],
                echo=False
            )
            
            result = response['choices'][0]['text'].strip()
            
            # Parse structured output if requested
            if structured_output:
                try:
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse structured output")
                    result = {"answer": result, "sources": [], "confidence": 0.5}
            
            # Prepare final response with metadata
            if include_metadata:
                final_response = {
                    'answer': result,
                    'metadata': {
                        'chunks_used': len(optimized_chunks),
                        'total_chunks': len(context_chunks),
                        'mode': mode.value,
                        'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                        'sources': [
                            {
                                'title': chunk.get('metadata', {}).get('title', 'Unknown'),
                                'id': chunk.get('id', ''),
                                'relevance': chunk.get('metadata', {}).get('relevance_score', 0.5)
                            }
                            for chunk in optimized_chunks
                        ],
                        'latency': time.time() - start_time
                    }
                }
            else:
                final_response = result
            
            # Update stats
            if 'usage' in response:
                self.stats['total_tokens'] += response['usage'].get('total_tokens', 0)
            
            # Cache successful response
            if self.enable_caching and self.cache_manager and not structured_output:
                self.cache_manager.set(cache_key, final_response)
            
            # Audit logging
            elapsed = time.time() - start_time
            LLM_LATENCY.labels(operation='rag_synthesis').observe(elapsed)
            self.audit_logger.log_request(
                user_id=user_id,
                query=query,
                model=self.model_profile.value,
                context_chunks=len(optimized_chunks),
                response_time=elapsed,
                success=True,
                metadata=metadata
            )
            
            return final_response
            
        except Exception as e:
            LLM_ERRORS.labels(type=type(e).__name__).inc()
            
            with self.metrics_lock:
                self.stats['errors'] += 1
                self.health_status['error_count'] += 1
            
            elapsed = time.time() - start_time
            LLM_LATENCY.labels(operation='rag_synthesis_error').observe(elapsed)
            self.audit_logger.log_request(
                user_id=user_id,
                query=query,
                model=self.model_profile.value,
                context_chunks=len(context_chunks),
                response_time=elapsed,
                success=False,
                metadata={'error': str(e)}
            )
            
            logger.error(f"RAG synthesis failed: {e}", exc_info=True)
            return self._get_fallback_response(query, e, context=True)
    
    def _get_system_prompt(self, mode: ResponseMode) -> str:
        """Get system prompt for specific mode"""
        prompts = {
            ResponseMode.CONCISE: (
                "You are a precise enterprise assistant. Answer using ONLY the provided context. "
                "If the answer isn't in the context, say 'This information is not available in the provided documents.' "
                "Keep responses brief and factual."
            ),
            ResponseMode.DETAILED: (
                "You are a comprehensive enterprise assistant. Provide detailed answers based on the context. "
                "If information is partially available, explain what you know and what's missing. "
                "Cite specific sections from the context when possible."
            ),
            ResponseMode.STRICT: (
                "You are a strict compliance assistant. Only use information explicitly stated in the context. "
                "Do not make inferences or add external knowledge. If information is missing, state that clearly."
            ),
            ResponseMode.ANALYTICAL: (
                "You are an analytical assistant. Analyze the provided context and provide insights. "
                "Compare and contrast information from different sources. Highlight key patterns and implications."
            ),
            ResponseMode.SUMMARIZATION: (
                "You are a summarization assistant. Create a concise summary of the key information from the context. "
                "Focus on the most important points and exclude redundant information."
            )
        }
        return prompts.get(mode, prompts[ResponseMode.CONCISE])
    
    def _handle_stream_response(self, response_iterator, cache_key: Optional[str] = None):
        """Handle streaming response with caching"""
        full_response = []
        
        for chunk in response_iterator:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                content = chunk['choices'][0].get('text', '')
                if content:
                    full_response.append(content)
                    yield content
        
        # Cache complete response
        if self.enable_caching and self.cache_manager and cache_key:
            self.cache_manager.set(cache_key, ''.join(full_response))
    
    def _get_fallback_response(self, query: str, error: Exception, context: bool = False) -> str:
        """Get graceful degradation response"""
        if context:
            return (
                "I apologize, but I'm unable to process your request with the provided context at this moment. "
                "Please try again later or contact support if the issue persists."
            )
        
        return "I'm currently experiencing technical difficulties. Please try again in a few moments."
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    if self.llm:
                        # Update health metrics
                        self.health_status['last_check'] = datetime.now().isoformat()
                        
                        # Check memory usage
                        process = psutil.Process()
                        mem_usage = process.memory_info().rss
                        LLM_MEMORY_USAGE.set(mem_usage)
                        
                        # Check if model is responsive
                        if self.stats['total_requests'] > 0:
                            avg_latency = self.stats['total_time'] / self.stats['total_requests']
                            self.health_status['avg_latency'] = avg_latency
                            
                            # Alert if latency too high
                            if avg_latency > 10:  # 10 seconds threshold
                                logger.warning(f"High average latency: {avg_latency:.2f}s")
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(300)  # Back off on error
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.metrics_lock:
            stats = dict(self.stats)
            stats['health_status'] = self.health_status
            stats['model_profile'] = self.model_profile.value
            stats['uptime'] = (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            stats['avg_tokens_per_request'] = (
                self.stats['total_tokens'] / max(self.stats['total_requests'], 1)
            )
            stats['error_rate'] = (
                self.stats['errors'] / max(self.stats['total_requests'], 1)
            )
            stats['cache_hit_rate'] = (
                self.stats['cache_hits'] / max(self.stats['total_requests'], 1)
            )
            return stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self.metrics_lock:
            self.stats = {
                'total_requests': 0,
                'total_tokens': 0,
                'total_time': 0.0,
                'errors': 0,
                'cache_hits': 0
            }
            self.health_status['error_count'] = 0
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        status = {
            'status': self.health_status['status'],
            'timestamp': datetime.now().isoformat(),
            'model': self.model_profile.value,
            'context_size': self.context_manager.max_context_tokens,
            'requests_total': self.stats['total_requests'],
            'errors_total': self.stats['errors'],
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        
        # Add detailed metrics if in monitoring mode
        if self.enable_monitoring:
            status.update({
                'cpu_percent': psutil.cpu_percent(interval=1),
                'available_memory': psutil.virtual_memory().available / 1024 / 1024,
                'disk_usage': psutil.disk_usage('/').percent,
                'avg_latency': self.health_status.get('avg_latency', 0)
            })
        
        return status
    
    def reload_model(self):
        """Reload model (useful for hot-swapping)"""
        logger.info("Reloading model...")
        self.llm = None
        self._initialize_model()
    
    def __enter__(self):
        """Context manager entry"""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self.llm:
            # Cleanup resources
            logger.info("Shutting down LLM service...")
            # Any necessary cleanup
    
    def __del__(self):
        """Destructor for cleanup"""
        logger.info("LLM service destroyed")

# Enterprise usage examples optimized for i5-4200U
if __name__ == "__main__":
    # Example 1: Basic usage with context manager
    with EnterpriseLLMService(
        model_profile=ModelProfile.CPU_OPTIMIZED,
        enable_caching=True,
        audit_level=AuditLevel.BASIC,
        enable_monitoring=False  # Disabled for performance
    ) as llm:
        
        # Simple query
        response = llm.generate_response(
            query="What is the return policy?",
            user_id="user123",
            mode=ResponseMode.CONCISE
        )
        print(f"Response: {response}")
        
        # RAG with context
        chunks = [
            {
                'id': 'doc1',
                'content': 'Our return policy allows returns within 30 days.',
                'metadata': {'title': 'Returns Policy', 'relevance_score': 0.9}
            },
            {
                'id': 'doc2',
                'content': 'Items must be in original condition with receipt.',
                'metadata': {'title': 'Return Conditions', 'relevance_score': 0.8}
            }
        ]
        
        rag_response = llm.synthesize_rag_response(
            query="Can I return items after 30 days?",
            context_chunks=chunks,
            mode=ResponseMode.STRICT,
            user_id="user123",
            include_metadata=True
        )
        print(f"RAG Response: {rag_response}")
        
        # Get stats
        stats = llm.get_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")