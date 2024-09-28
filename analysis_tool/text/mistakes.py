from openai.types.audio import TranscriptionVerbose

from analysis_tool.mistakes.mistakes import MistakeType
from analysis_tool.mistakes.models import Mistake

LONG_PAUSE_THRESHOLD = 2


def find_pauses(transcription: TranscriptionVerbose) -> list[Mistake]:
    if len(transcription.words) < 2:
        return []

    mistakes = []
    for previous_word, next_word in zip(transcription.words[:-1], transcription.words[1:]):
        if next_word.start - previous_word.end > LONG_PAUSE_THRESHOLD:
            mistakes.append(
                Mistake(
                    type=MistakeType.PAUSING,
                    confidence=1,
                    start_ts=previous_word.end,
                    end_ts=next_word.start,
                )
            )

    return mistakes
