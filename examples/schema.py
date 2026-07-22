from typing import Literal

from pydantic import BaseModel, ConfigDict


class SupportAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["billing", "technical", "account", "other"]
    answer: str
    should_escalate: bool
