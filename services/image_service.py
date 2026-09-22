from pathlib import Path


def ensure_dir(path: str) -> str:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)
