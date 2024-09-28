from enum import StrEnum


class MistakeType(StrEnum):
    # ============== TEXT =================
    # Searching for single character repetitions in single string like `yyy` or `eee`
    INTERLUDES = "interludes"

    # 140 -160 words per minute is good tempo, i would set too_slow <= 135, too_fast >= 165
    SPEAKING_TOO_FAST = "speaking too fast"  # TODO: na potem moze ogarnac jak to zrobic na pojedynczych zdaniach

    # Just detecting repetition in words ???
    REPETITIONS = "repetitions"
    CHANGING_TOPIC = "changing the topic of speech"

    # Numbers per minute ? Per sentence ? Should be calculated dynamically for frames
    TOO_MANY_NUMBERS = "too many numbers"
    LONG_DIFFICULT_WORDS = "too long, difficult words, sentences"
    JARGON = "jargon"
    FOREIGN_LANGUAGE = "foreign language"
    # Some hard core AI model
    FALSE_WORDS = "false words"
    PASSIVE_SIDE = "use of the passive side, e.g. given, indicated, summarized"

    # ============== AUDIO =================

    # Search for too high decibels in wav file. 70-80db is perfect speech volume (acc. to internet)
    SPEAKING_LOUD = "speaking louder"
    SPEAKING_QUIETLY = "speaking too quietly, in a whisper"
    NOISE = "noise"
    ACCENTUATION = "accentuation"

    # Here we have to search for over 5 second low dB values in wav file
    PAUSING = "pausing too long"

    # ============== VIDEO =================

    # Face recognition, there should be only one face (big face, can be more smaller ones but one big xD)
    SECOND_PLAN_PERSON = "second plan - another person on the set"
    TURNING_AWAY = "turning away, twisting, gesticulating"
    FACIAL_EXPRESSIONS = "facial expressions"
    INCONSISTENT_TRANSCRIPT = "inconsistent speech with the transcript"
