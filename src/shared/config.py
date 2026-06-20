from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "IR Research Assistant"
    data_dir: Path = Path("data")
    log_level: str = "INFO"


settings = Settings()
