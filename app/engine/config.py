"""
Configuration management with Low Resource Mode toggle.
All optimizations can be enabled/disabled via config.
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum

class ResourceMode(str, Enum):
    LOW = "low"  # CPU-optimized for i5-4200U
    STANDARD = "standard"  # Balanced
    HIGH = "high"  # Full enterprise features

class Settings(BaseSettings):
    """
    Centralized configuration with Low Resource Mode.
    All optimizations can be controlled via this config.
    """
    
    # ============================================================================
    # Core Settings
    # ============================================================================
    APP_NAME: str = "EnterpriseRAG"
    ENVIRONMENT: str = "production"
    RESOURCE_MODE: ResourceMode = ResourceMode.LOW  # Default to LOW for i5-4200U
    
    # ============================================================================
    # Low Resource Mode Optimizations (Toggle Everything Here)
    # ============================================================================
    @property
    def IS_LOW_RESOURCE(self) -> bool:
        """Check if we're in low resource mode"""
        return self.RESOURCE_MODE == ResourceMode.LOW
    
    # ============================================================================
    # Model Settings
    # ============================================================================
    MODEL_PATH: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    MODEL_THREADS: int = Field(2, description="Thread count for LLM")
    MODEL_BATCH_SIZE: int = Field(256, description="Batch size for inference")
    MODEL_CONTEXT_SIZE: int = Field(1024, description="Context window size")
    
    @validator('MODEL_THREADS', pre=True, always=True)
    def validate_threads(cls, v, values):
        """Auto-optimize threads based on resource mode"""
        if values.get('RESOURCE_MODE') == ResourceMode.LOW:
            return 2  # Optimal for i5-4200U
        return v
    
    # ============================================================================
    # Intent Classification Settings
    # ============================================================================
    INTENT_USE_TRANSFORMER: bool = Field(False, description="Use transformer model (heavy)")
    INTENT_USE_SPACY: bool = Field(False, description="Use spaCy (medium)")
    INTENT_USE_ML: bool = Field(True, description="Use lightweight ML")
    INTENT_USE_REGEX: bool = Field(True, description="Use regex patterns")
    INTENT_MODEL_PATH: str = Field("models/intent_model.joblib", description="Path to intent ML model")
    INTENT_CACHE_SIZE: int = Field(500, description="Intent cache size")
    
    @validator('INTENT_USE_TRANSFORMER', 'INTENT_USE_SPACY', pre=True, always=True)
    def validate_intent_settings(cls, v, values):
        """Auto-disable heavy models in low resource mode"""
        if values.get('RESOURCE_MODE') == ResourceMode.LOW:
            return False  # Disable heavy models
        return v
    
    # ============================================================================
    # Cache Settings (Optimized for 8GB RAM)
    # ============================================================================
    CACHE_MEMORY_MAXSIZE: int = Field(500, description="Memory cache size")
    CACHE_MEMORY_TTL: int = Field(300, description="Cache TTL in seconds")
    CACHE_REDIS_URL: Optional[str] = Field(None, description="Redis URL")
    
    @validator('CACHE_MEMORY_MAXSIZE', pre=True, always=True)
    def validate_cache_size(cls, v, values):
        """Reduce cache size in low resource mode"""
        if values.get('RESOURCE_MODE') == ResourceMode.LOW:
            return 300  # Smaller cache for 8GB RAM
        return v
    
    # ============================================================================
    # Retrieval Settings
    # ============================================================================
    RETRIEVAL_TOP_K: int = Field(5, description="Number of docs to retrieve")
    RETRIEVAL_DENSE_K: int = Field(3, description="Dense retrieval count")
    RETRIEVAL_KEYWORD_K: int = Field(2, description="Keyword retrieval count")
    RETRIEVAL_STRATEGIES: list = Field(
        default=["dense", "keyword"],  # Remove hybrid by default in low resource
        description="Active retrieval strategies"
    )
    
    @validator('RETRIEVAL_STRATEGIES', pre=True, always=True)
    def validate_strategies(cls, v, values):
        """Simplify strategies in low resource mode"""
        if values.get('RESOURCE_MODE') == ResourceMode.LOW:
            return ["dense", "keyword"]  # Remove hybrid (too heavy)
        return v
    
    # ============================================================================
    # Background Task Settings
    # ============================================================================
    MONITORING_ENABLED: bool = Field(False, description="Enable monitoring")
    MONITORING_INTERVAL: int = Field(300, description="Monitoring interval (seconds)")
    METRICS_COLLECTION_INTERVAL: int = Field(300, description="Metrics interval")
    
    @validator('MONITORING_ENABLED', pre=True, always=True)
    def validate_monitoring(cls, v, values):
        """Disable monitoring in low resource mode"""
        if values.get('RESOURCE_MODE') == ResourceMode.LOW:
            return False  # Save CPU cycles
        return v
    
    # ============================================================================
    # Rate Limiting
    # ============================================================================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = Field(100, description="Requests per window")
    RATE_LIMIT_WINDOW: int = Field(60, description="Window in seconds")
    
    # ============================================================================
    # Circuit Breaker
    # ============================================================================
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_THRESHOLD: int = Field(5, description="Failure threshold")
    CIRCUIT_BREAKER_TIMEOUT: int = Field(60, description="Recovery timeout")
    
    # ============================================================================
    # A/B Testing
    # ============================================================================
    AB_TESTING_ENABLED: bool = Field(True, description="Enable A/B testing")
    AB_TEST_GROUPS: list = Field(default=["control", "test_a", "test_b"])
    
    # ============================================================================
    # Token Counting
    # ============================================================================
    TOKEN_COUNT_METHOD: str = Field(
        "llama",  # Use llama tokenizer when available
        description="Token counting method: llama, tiktoken, or approximate"
    )
    
    # ============================================================================
    # Logging & Tracing
    # ============================================================================
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    ENABLE_TRACING: bool = Field(False, description="Enable OpenTelemetry")
    TRACE_SAMPLE_RATE: float = Field(0.1, description="Tracing sample rate")
    
    # ============================================================================
    # LangGraph Settings
    # ============================================================================
    LANGGRAPH_CHECKPOINT_ENABLED: bool = Field(True, description="Enable checkpointing")
    LANGGRAPH_MAX_HISTORY: int = Field(10, description="Max conversation history")
    MEMORY_WINDOW_TOKENS: int = Field(512, description="Max tokens for history window")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()