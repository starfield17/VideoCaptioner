"""Typed errors shared by the Phase 0 application boundary."""


class CaptionerError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(CaptionerError):
    """Configuration or explicit input validation failed."""


class MediaPreparationError(CaptionerError):
    """A media fixture could not be prepared."""


class ProviderUnavailableError(CaptionerError):
    """The selected ASR provider could not be started."""


class TranscriptionError(CaptionerError):
    """The ASR worker returned an invalid or failed result."""


class AlignmentError(CaptionerError):
    """A forced aligner could not produce a valid timed transcript."""


class LlmAuthenticationError(CaptionerError):
    """An LLM adapter was not authenticated."""


class LlmPermanentError(CaptionerError):
    """An LLM request failed permanently."""


class LlmRetryableError(CaptionerError):
    """A bounded LLM retry policy exhausted a transient request."""


class StructuredOutputError(CaptionerError):
    """An LLM response did not satisfy its structured schema."""


class SubtitleValidationError(CaptionerError):
    """A subtitle stage returned an invalid document or batch."""


class ExportError(CaptionerError):
    """A subtitle artifact could not be written."""
