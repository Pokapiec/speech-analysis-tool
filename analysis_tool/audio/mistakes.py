from analysis_tool.audio.audio_parser import AudioParser
from analysis_tool.audio.volume_analyzer import AudioVolumeAnalyzer
from analysis_tool.mistakes.models import Mistake
from analysis_tool.mistakes.mistakes import MistakeCategory, MistakeType


def get_audio_mistakes(audio: AudioParser) -> list[Mistake]:
    return []


def get_volume_mistakes(volume: AudioVolumeAnalyzer) -> list[Mistake]:
    mistakes = []

    data_to_capture = [
        (MistakeType.SPEAKING_LOUD, volume.get_too_loud_fragments),
        (MistakeType.SPEAKING_LOUD, volume.get_high_noise_fragments),
        (MistakeType.SPEAKING_LOUD, volume.get_too_quiet_fragments),
    ]
    for mistake_type, generate_mistakes_func in data_to_capture:
        data = generate_mistakes_func()
        for start_ts, end_ts in data:
            mistakes.append(
                Mistake(mistake_type, MistakeCategory.AUDIO, confidence=1, start_ts=start_ts, end_ts=end_ts)
            )

    return mistakes
