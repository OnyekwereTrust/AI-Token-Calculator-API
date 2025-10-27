import os
from typing import List, Optional
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    
    port: int = 8000
    pricing_url: Optional[str] = None
    admin_api_key: str = "change-me"
    cors_origins: str = "*"
    log_level: str = "INFO"
    
    @classmethod
    def load(cls):
        """Load settings from environment variables."""
        return cls(
            port=int(os.getenv("PORT", "8000")),
            pricing_url=os.getenv("PRICING_URL"),
            admin_api_key=os.getenv("ADMIN_API_KEY", "change-me"),
            cors_origins=os.getenv("CORS_ORIGINS", "*"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


settings = Settings.load()
