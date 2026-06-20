import logging

from src.shared.config import settings


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=settings.log_level)
    return logging.getLogger(name)
