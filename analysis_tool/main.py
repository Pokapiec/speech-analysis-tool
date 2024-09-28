import ssl

from analysis_tool.audio.audio_parser import AudioParser
from analysis_tool.audio.mistakes import get_audio_mistakes
from analysis_tool.text.mistakes import get_text_mistakes
from analysis_tool.video.mistakes import get_video_mistakes
from analysis_tool.video.video_parser import VideoParser

ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    video = VideoParser("szybkie_tempo__drugi_plan__mowienie_glosniej.mp4")
    audio_name = video.save_mp3()
    audio = AudioParser(audio_name)

    text_mistakes = get_text_mistakes(audio.transcript)
    audio_mistakes = get_audio_mistakes(audio)
    video_mistakes = get_video_mistakes(video)

    mistakes = [*text_mistakes, *audio_mistakes, *video_mistakes]
