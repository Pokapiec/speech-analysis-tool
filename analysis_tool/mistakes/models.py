from dataclasses import dataclass

from analysis_tool.mistakes.mistakes import MistakeType


@dataclass
class Mistake:
    type: MistakeType
    confidence: float  # 0-100%
    start_ts: float
    end_ts: float | None = None
