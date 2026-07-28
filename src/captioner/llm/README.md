# LLM

The public `CloudLlm` contract supports Boundary-ID segmentation, correction,
translation, and one repair stage. Phase 2 adds the strict OpenAI-compatible
adapter, thread-local clients, bounded retries, and the single
`ParallelLlmExecutor`; `FakeLlm` remains the offline test provider.
