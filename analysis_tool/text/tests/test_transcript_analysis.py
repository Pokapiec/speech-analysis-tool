from hamcrest import assert_that, contains_exactly, has_properties
from openai.types.audio import TranscriptionVerbose, TranscriptionWord

from analysis_tool.mistakes.mistakes import MistakeType
from analysis_tool.text.mistakes import find_pauses
from analysis_tool.text.text_errors_parser import TextErrorsParser


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


def test_calculate_speech_pace():
    transcript_speech_pace = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="Text without some pauses",
        segments=None,
        words=[
            TranscriptionWord(start=6, end=7, word="Text"),
            TranscriptionWord(start=8, end=9, word="with"),
            TranscriptionWord(start=13, end=14, word="some"),
            TranscriptionWord(start=15, end=16, word="pauses"),
        ],
        task="transcribe",
    )

    pace = TextErrorsParser(transcript_speech_pace).calculate_speech_pace()
    assert pace == 37.5


def test_detect_repetitions():
    transcript_repetitions = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="Text without some texts",
        segments=None,
        words=[
            TranscriptionWord(start=6, end=7, word="Text"),
            TranscriptionWord(start=8, end=9, word="with"),
            TranscriptionWord(start=13, end=14, word="some"),
            TranscriptionWord(start=15, end=16, word="texts"),
        ],
        task="transcribe",
    )

    repetitions = TextErrorsParser(transcript_repetitions).detect_repetitions()
    assert set(repetitions) == {"Text", "texts"}


def test_count_numbers_per_sentence():
    transcript_numbers = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="The 15th quoter 15%. That's it.",
        segments=None,
        words=[
            TranscriptionWord(start=6, end=7, word="The"),
            TranscriptionWord(start=8, end=9, word="15th"),
            TranscriptionWord(start=13, end=14, word="quoter"),
            TranscriptionWord(start=15, end=16, word="15%"),
            TranscriptionWord(start=15, end=16, word="That's"),
            TranscriptionWord(start=15, end=16, word="it"),
        ],
        task="transcribe",
    )

    number_per_sentence = TextErrorsParser(transcript_numbers).count_numbers_per_sentence()
    assert number_per_sentence == [2, 0]


def test_text_fog_index():
    transcript_numbers = TranscriptionVerbose(
        duration="20",
        language="polish",
        text="FOG index for complex sentence to test it",
        segments=None,
        words=[
            TranscriptionWord(start=6, end=7, word="FOG"),
            TranscriptionWord(start=8, end=9, word="index"),
            TranscriptionWord(start=13, end=14, word="for"),
            TranscriptionWord(start=15, end=16, word="complex"),
            TranscriptionWord(start=17, end=18, word="sentence"),
            TranscriptionWord(start=19, end=20, word="to"),
            TranscriptionWord(start=21, end=22, word="test"),
            TranscriptionWord(start=23, end=24, word="it"),
        ],
        task="transcribe",
    )

    fog_index = TextErrorsParser(transcript_numbers).calculate_fog_index()
    assert round(fog_index, 2) == 3.2