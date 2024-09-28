import ssl

from analysis_tool.audio.audio_parser import AudioParser
from analysis_tool.video.video_parser import VideoParser

ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    video = VideoParser("szybkie_tempo__drugi_plan__mowienie_glosniej.mp4")
    audio_name = video.save_mp3()
    audio = AudioParser(audio_name)
    print()
