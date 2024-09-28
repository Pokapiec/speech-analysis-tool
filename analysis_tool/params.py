from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Envs:
    OPENAPI_KEY: str


def load_envs() -> Envs:
    envs = {}
    with open(f"{PROJECT_ROOT}/.env", "r") as f:
        for line in f:
            key, value = line.split("=", 1)
            envs[key] = value

    return envs
