class IRResearchAssistantError(Exception):
    """Base exception for application errors."""


class PaperNotFoundError(IRResearchAssistantError):
    """Raised when a requested paper cannot be found."""
