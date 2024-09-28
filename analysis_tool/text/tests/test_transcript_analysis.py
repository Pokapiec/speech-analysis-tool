from hamcrest import assert_that, contains_exactly, has_properties
from openai.types.audio import TranscriptionVerbose, TranscriptionWord

from analysis_tool.mistakes.mistakes import MistakeType
from analysis_tool.text.mistakes import find_pauses


def test_find_pauses():
    # given
    transcript_with_long_pause = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="Text with long pause",
        segments=None,
        words=[
            TranscriptionWord(end=7, start=6, word="Text"),
            TranscriptionWord(end=9, start=8, word="with"),
            TranscriptionWord(end=14, start=13, word="long"),
            TranscriptionWord(end=16, start=15, word="pause"),
        ],
        task="transcribe",
    )

    # when
    mistakes = find_pauses(transcript_with_long_pause)

    # then
    assert_that(
        mistakes,
        contains_exactly(
            has_properties({"type": MistakeType.PAUSING, "start_ts": 9, "end_ts": 13})
        ),
    )


def test_find_pauses_no_mistakes():
    # given
    transcript_with_long_pause = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="Text without pause",
        segments=None,
        words=[
            TranscriptionWord(end=7, start=6, word="Text"),
            TranscriptionWord(end=9, start=8, word="with"),
            TranscriptionWord(end=15, start=10, word="pause"),
        ],
        task="transcribe",
    )

    # when
    mistakes = find_pauses(transcript_with_long_pause)

    # then
    assert not mistakes
