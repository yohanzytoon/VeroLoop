# Privacy and data retention

Evaluation inputs, system prompts, and generated outputs may contain sensitive data. Real adapters
send them to their provider, where that provider's account settings, terms, region, and retention
policy apply. EvalFrame does not host or retain data itself.

JSON reports intentionally preserve outputs and errors for reproducibility. Store them with suitable
access control and retention. Errors are sanitized for common credential forms, but callers should
still avoid putting secrets into prompts or metadata. Candidate prompts are omitted from reports.
MLflow logs parameters, hashes, summaries, and a content-redacted artifact by default. Enabling
`log_content=True` is an explicit choice to log outputs to the configured tracking store.

