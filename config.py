"""
PromptOS Configuration - Central configuration management.

This module handles all configuration settings for the PromptOS system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///promptos.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

@dataclass
class RedisConfig:
    """Redis configuration for caching."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 100

@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.3
    timeout: int = 30

@dataclass
class AgentConfig:
    """Agent configuration settings."""
    max_concurrent_tasks: int = 10
    task_timeout: int = 300
    retry_attempts: int = 3
    enable_learning: bool = True
    enable_caching: bool = True

@dataclass
class SecurityConfig:
    """Security and compliance settings."""
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_encryption: bool = True
    allowed_origins: list = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]

@dataclass
class MonitoringConfig:
    """Monitoring and observability settings."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    metrics_port: int = 9090
    enable_dashboards: bool = True

class PromptOSConfig:
    """Main configuration class for PromptOS."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.openai = OpenAIConfig()
        self.agents = AgentConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        
        # Project paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.cache_dir = self.project_root / "cache"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        for directory in [self.data_dir, self.logs_dir, self.cache_dir]:
            directory.mkdir(exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("ENVIRONMENT", "development") == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development") == "production"

# Global configuration instance
config = PromptOSConfig()
