import difflib
import string

from openai.types.audio import TranscriptionVerbose

from analysis_tool.audio.openai_api import recognize_passive_voice_words
from analysis_tool.mistakes.mistakes import MistakeType, MistakeCategory
from analysis_tool.mistakes.models import Mistake

LONG_PAUSE_THRESHOLD = 2


def get_text_mistakes(transcription: TranscriptionVerbose) -> list[Mistake]:
    pauses = find_pauses(transcription)
    passive_voice_mistakes = find_passive_voice(transcription)

    return [*pauses, *passive_voice_mistakes]


def find_pauses(transcription: TranscriptionVerbose) -> list[Mistake]:
    if len(transcription.words) < 2:
        return []

    mistakes = []
    for previous_word, next_word in zip(
        transcription.words[:-1], transcription.words[1:]
    ):
        if next_word.start - previous_word.end > LONG_PAUSE_THRESHOLD:
            mistakes.append(
                Mistake(
                    type=MistakeType.PAUSING,
                    category=MistakeCategory.TEXT,
                    confidence=1,
                    start_ts=previous_word.end,
                    end_ts=next_word.start,
                )
            )

    return mistakes


def compare_transcription(transcription: str, subtitles: str) -> list[Mistake]:
    similarity_ratio = difflib.SequenceMatcher(
        None, clean_string(transcription), clean_string(subtitles)
    ).ratio()
    threshold = 0.98

    if similarity_ratio < threshold:
        confidence = max(similarity_ratio**3 - 0.5, 0)
        return [
            Mistake(
                type=MistakeType.INCONSISTENT_TRANSCRIPT,
                category=MistakeCategory.VIDEO,
                start_ts=0,
                confidence=confidence,
            )
        ]


def clean_string(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()


def find_passive_voice(transcription: TranscriptionVerbose) -> list[Mistake]:
    passive_voice_words = recognize_passive_voice_words(transcription)

    return [
        Mistake(
            type=MistakeType.PASSIVE_SIDE,
            category=MistakeCategory.TEXT,
            confidence=1,
            start_ts=word.start,
            end_ts=word.end,
        )
        for word in passive_voice_words
    ]
