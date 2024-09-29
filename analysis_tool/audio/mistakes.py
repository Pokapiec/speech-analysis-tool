from analysis_tool.audio.audio_parser import AudioParser
from analysis_tool.audio.volume_analyzer import AudioVolumeAnalyzer
from analysis_tool.mistakes.mistakes import MistakeCategory, MistakeType
from analysis_tool.mistakes.models import Mistake


def get_audio_mistakes(audio: AudioParser) -> list[Mistake]:
    volume_analyzer = AudioVolumeAnalyzer(audio.file_name)

    volume_mistakes = get_volume_mistakes(volume_analyzer)
    return volume_mistakes


def get_volume_mistakes(volume: AudioVolumeAnalyzer) -> list[Mistake]:
    mistakes = []

    data_to_capture = [
        (MistakeType.SPEAKING_LOUD, volume.get_too_loud_fragments),
        (MistakeType.NOISE, volume.get_high_noise_fragments),
        (MistakeType.SPEAKING_QUIETLY, volume.get_too_quiet_fragments),
    ]
    for mistake_type, generate_mistakes_func in data_to_capture:
        data = generate_mistakes_func()
        for start_ts, end_ts in data:
            mistakes.append(
                Mistake(
                    mistake_type,
                    MistakeCategory.AUDIO,
                    confidence=1,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )

    return mistakes
