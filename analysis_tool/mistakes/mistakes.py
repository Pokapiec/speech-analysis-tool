from enum import StrEnum


class Mistake(StrEnum):
    INTERLUDES = "interludes"
    SPEAKING_TOO_FAST = "speaking too fast"
    REPETITIONS = "repetitions"
    CHANGING_TOPIC = "changing the topic of speech"
    TOO_MANY_NUMBERS = "too many numbers"
    LONG_DIFFICULT_WORDS = "too long, difficult words, sentences"
    JARGON = "jargon"
    FOREIGN_LANGUAGE = "foreign language"
    PAUSING = "pausing too long"
    SPEAKING_LOUD = "speaking louder"
    SPEAKING_QUIETLY = "speaking too quietly, in a whisper"
    SECOND_PLAN_PERSON = "second plan - another person on the set"
    TURNING_AWAY = "turning away, twisting, gesticulating"
    FACIAL_EXPRESSIONS = "facial expressions"
    FALSE_WORDS = "false words"
    INCONSISTENT_TRANSCRIPT = "inconsistent speech with the transcript"
    NOISE = "noise"
    PASSIVE_SIDE = "use of the passive side, e.g. given, indicated, summarized"
    ACCENTUATION = "accentuation"
