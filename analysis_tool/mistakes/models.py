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

    def __str__(self):
        return f"{self.type} @ {round(self.start_ts, 1)}s - {round(self.end_ts or self.start_ts + 1, 1)}s (confidence: {round(self.confidence, 2)})"
