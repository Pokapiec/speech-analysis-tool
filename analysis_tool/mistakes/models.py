from dataclasses import dataclass

from analysis_tool.mistakes.mistakes import MistakeType, MistakeCategory


@dataclass
class Mistake:
    type: MistakeType
    category: MistakeCategory
    confidence: float  # 0-100%
    start_ts: float
    end_ts: float | None = None
    detail: str | None = None
