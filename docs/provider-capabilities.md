# Provider capabilities

Capabilities are declared separately from candidate model names. The current OpenAI and Gemini
adapters select native JSON Schema output; Anthropic selects a required tool schema; the fake
adapter supports both for offline tests. These are adapter capabilities, not promises that every
model/version supports every JSON Schema keyword. Users must choose compatible model IDs.

Version 0.1 rejects any keyword outside a supplied capability allow-list. The built-in adapters do
not yet ship model-specific allow-lists because those rapidly changing claims belong in a versioned
registry. No prompt-only or JSON-mode fallback is automatic.

