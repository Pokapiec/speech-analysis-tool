import os

from openai.types.audio import TranscriptionVerbose

from analysis_tool.audio.openai_api import generate_transcript_from_mp3
from analysis_tool.params import AUDIO_FILES_PATH


class AudioParser:
    def __init__(self, file_name: str):
        self.file_name: str = file_name
        self.file_path: str = os.path.join(AUDIO_FILES_PATH, file_name)
        self.transcript: TranscriptionVerbose = self.extract_transcript()

    def extract_transcript(self) -> TranscriptionVerbose:
        return generate_transcript_from_mp3(self.file_name)
