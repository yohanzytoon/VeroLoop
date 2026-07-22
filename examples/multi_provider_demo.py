"""Sketch real-provider setup while reusing the same task and candidates."""

import os


def main() -> None:
    required = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "Set provider credentials (missing: "
            + ", ".join(missing)
            + ") and install: uv sync --extra providers"
        )
    raise SystemExit("Configure model IDs and instantiate the adapters documented in README.md.")


if __name__ == "__main__":
    main()
