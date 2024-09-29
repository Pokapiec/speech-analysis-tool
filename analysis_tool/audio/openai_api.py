import ast
import os
import pickle

from openai import OpenAI
from openai.types.audio import TranscriptionVerbose, TranscriptionWord

from analysis_tool.params import load_envs, PROJECT_ROOT, AUDIO_FILES_PATH

TRANSCRIPT_CACHE_PATH = os.path.join(
    PROJECT_ROOT, "analysis_tool", "audio", "transcript_cache"
)


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


def prompt_gpt(prompt: str, max_tokens: int = 500):
    client = OpenAI(api_key=get_openapi_key())
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=max_tokens,
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content.strip()


def recognize_passive_voice_words(
    transcript: TranscriptionVerbose,
) -> list[TranscriptionWord]:
    prompt = f"""Z tekstu zwróć wszystkie czasowniki w formie biernej w języku polskim w formacie 
    '```python["słowo_1", "słowo_2", ...]```'. 
    Zwracaj tylko pojedyncze słowa, używając minimalnej liczby znaków: {transcript.text}"""

    passive_words = prompt_gpt(prompt)
    list_string = passive_words.replace("python", "").replace("`", "").strip()
    detected_words = set(ast.literal_eval(list_string))
    return list(filter(lambda x: x.word in detected_words, transcript.words))
