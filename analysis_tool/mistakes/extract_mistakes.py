from analysis_tool.audio.audio_parser import AudioParser
from analysis_tool.audio.mistakes import get_audio_mistakes
from analysis_tool.mistakes.models import Mistake
from analysis_tool.text.mistakes import get_text_mistakes, compare_transcription
from analysis_tool.video.mistakes import get_video_mistakes
from analysis_tool.video.video_parser import VideoParser


def extract_mistakes_from_video(file_name: str) -> list[Mistake]:
    video = VideoParser(file_name)
    audio_name = video.save_mp3()
    audio = AudioParser(audio_name)

    text_mistakes = get_text_mistakes(audio.transcript)
    audio_mistakes = get_audio_mistakes(audio)
    video_mistakes = get_video_mistakes(video)
    transcription_mistakes = compare_transcription(
        transcription=audio.transcript.text, subtitles=video.ocr_subtitles
    )

    return [*text_mistakes, *audio_mistakes, *video_mistakes, *transcription_mistakes]
