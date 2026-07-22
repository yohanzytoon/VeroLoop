# Architecture

EvalFrame is Python-first because Pydantic models are executable user contracts and the evaluation
ecosystem is predominantly Python. The lightweight core contains domain models, schema handling,
metrics, orchestration, and reporters. Provider SDKs and MLflow are extras and are imported only
when their adapters are instantiated.

Tasks describe cases and their output contract. Candidates describe model, prompt, and parameters.
Adapters alone translate the provider-neutral `ModelRequest` and normalize SDK responses into a
`ModelResponse`; SDK objects never enter reports. The canonical schema compiler first creates a
stable Pydantic JSON Schema, selects strict native or tool output using declared capabilities, and
rejects unsupported semantics. Later compilers may add explicit transformations, but Version 0.1
does not silently degrade a schema.

The runner loads and validates cases, creates candidate/case/repetition jobs, enforces global and
per-provider semaphores, retries only normalized retryable provider failures, validates output,
scores attempts, sorts results deterministically, and aggregates summaries. Cancellation naturally
propagates through `asyncio.gather`; each adapter must remain cancellation-safe. Dependency injection
for adapters, prices, metrics, progress, and configuration makes the core testable offline.

Reporters consume a complete typed report. JSON is the durable interchange; Rich is presentation.
MLflow is an optional experiment integration rather than a product database. This keeps ownership,
retention, and deployment with the user.

Future baseline snapshots can reference the same report schema, while temporal replay can inject
frozen retrieval/tool responses at the request boundary. Tuning can produce candidates and consume
summaries. Neither concern needs to contaminate the evaluation runner.

