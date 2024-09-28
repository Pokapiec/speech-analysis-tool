import os

from openai import OpenAI

import pickle
from analysis_tool.params import load_envs, PROJECT_ROOT

VIDEO_FILES_PATH = "./video_files"
AUDIO_FILES_PATH = "./audio_files"


class OpenAISpeechToText:
    def __init__(self) -> None:
        self.openapi_key = load_envs()["OPENAPI_KEY"]

    def _add_to_cache(self, file_name: str, value: any):
        with open(f"{PROJECT_ROOT}/analysis_tool/audio/transcript_cache/{file_name}.p", "wb") as f:
            pickle.dump(value, f)
    
    def _load_from_cache(self, file_name: str) -> any:
        path = f"{PROJECT_ROOT}/analysis_tool/audio/transcript_cache/{file_name}.p"
        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    def mp4_to_mp3(self, file_name: str) -> str:
        """We need to cast mp4 files to wav to make it compatible with our speech recognition lib."""
        # -y flag is to always override files
        command2mp3 = f"ffmpeg -y -i {VIDEO_FILES_PATH}/{file_name}.mp4 {AUDIO_FILES_PATH}/{file_name}_speech.mp3"
        os.system(command2mp3)

        return f"{file_name}_speech.mp3"

    def generate_transcription(self, file_name: str):
        cached = self._load_from_cache(file_name)
        if cached is not None:
            print(f"{cached = }")
            return cached

        mp3_file_name = self.mp4_to_mp3(file_name)

        with open(f"{AUDIO_FILES_PATH}/{mp3_file_name}", "rb") as mp3_file:
            client = OpenAI(api_key=self.openapi_key)

            print("HITTING OPENAI")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=mp3_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="pl",
            )
            self._add_to_cache(file_name, transcript)
            print(f"{transcript = }")
