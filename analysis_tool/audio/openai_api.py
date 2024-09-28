import os
import pickle

from openai import OpenAI
from openai.types.audio import TranscriptionVerbose

from analysis_tool.params import load_envs, PROJECT_ROOT, AUDIO_FILES_PATH

TRANSCRIPT_CACHE_PATH = os.path.join(PROJECT_ROOT, "analysis_tool", "audio", "transcript_cache")


def get_openapi_key() -> str:
    return load_envs().OPENAPI_KEY


def _add_to_cache(file_name: str, value: TranscriptionVerbose) -> None:
    with open(os.path.join(TRANSCRIPT_CACHE_PATH, f"{file_name}.p"), "wb") as f:
        pickle.dump(value, f)


def _load_from_cache(file_name: str) -> TranscriptionVerbose | None:
    path = os.path.join(TRANSCRIPT_CACHE_PATH, f"{file_name}.p")

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def generate_transcript_from_mp3(file_name: str) -> TranscriptionVerbose:
    cached = _load_from_cache(file_name)
    if cached is not None:
        return cached

    with open(os.path.join(AUDIO_FILES_PATH, file_name), "rb") as mp3_file:
        client = OpenAI(api_key=get_openapi_key())

        print("HITTING OPENAI")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=mp3_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="pl",
        )
        _add_to_cache(file_name, transcript)
    return transcript
