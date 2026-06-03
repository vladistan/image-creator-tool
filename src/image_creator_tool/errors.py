"""Exception hierarchy for image-creator-tool.

Separates transient (retryable) from permanent (fatal) API failures
to enable intelligent retry logic in provider implementations.
"""


class ImageCreatorError(Exception):
    """Base exception for all image-creator-tool errors."""


class TransientAPIError(ImageCreatorError):
    """Retryable API failure (rate limits, server errors, network issues)."""


class PermanentAPIError(ImageCreatorError):
    """Non-retryable API failure (auth, content safety, invalid request)."""
