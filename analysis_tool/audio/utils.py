import os

VIDEO_FILES_PATH = "./video_files"
WAV_FILES_PATH = "./audio_files"


def mp4_to_wav(file_name: str) -> str:
    """We need to cast mp4 files to wav to make it compatible with our speech recognition lib."""
    # -y flag is to always override files
    command2mp3 = f"ffmpeg -y -i {VIDEO_FILES_PATH}/{file_name}.mp4 {WAV_FILES_PATH}/{file_name}_speech.mp3"
    command2wav = f"ffmpeg -y -i {WAV_FILES_PATH}/{file_name}_speech.mp3 {WAV_FILES_PATH}/{file_name}_speech.wav"

    os.system(command2mp3)
    os.system(command2wav)

    return f"{file_name}_speech.wav"
