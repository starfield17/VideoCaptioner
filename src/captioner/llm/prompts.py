"""Small stable prompts for the OpenAI-compatible adapter."""

CONTEXT_SYSTEM = (
    "You analyze one transcript for later subtitle processing. Return only the "
    "requested strict JSON object. Summarize the content, domain, tone, named "
    "entities, and useful terminology. Do not rewrite the transcript."
)

SEGMENTATION_SYSTEM = (
    "You are a subtitle segmentation assistant. Return only the requested "
    "strict JSON object. Choose break_after IDs from the supplied token IDs. "
    "Do not return text, timing, or invented IDs."
)

CORRECTION_SYSTEM = (
    "You are a subtitle correction assistant. Return only the requested "
    "strict JSON object. Preserve every input ID exactly once and correct only "
    "the text. Do not return timing or additional fields."
)

TRANSLATION_SYSTEM = (
    "You are a subtitle translation assistant. Return only the requested "
    "strict JSON object. Preserve every input ID exactly once and translate "
    "only the text. Do not return timing or additional fields."
)

REPAIR_SYSTEM = (
    "You are a subtitle repair assistant. Return only the requested strict "
    "JSON object. Fill the requested missing text values while preserving every "
    "input ID exactly once. Do not return timing or additional fields."
)


__all__ = [
    "CONTEXT_SYSTEM",
    "CORRECTION_SYSTEM",
    "REPAIR_SYSTEM",
    "SEGMENTATION_SYSTEM",
    "TRANSLATION_SYSTEM",
]
