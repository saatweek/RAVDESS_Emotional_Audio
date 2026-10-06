"""Download the Kaggle speech dataset and save the selected dataset directory.

This is data acquisition only. Read records() in ravdess.py next to understand
which filenames become labeled examples. See CODE_WALKTHROUGH.md.
"""
import os
from pathlib import Path


def main():
    """Set the cache default before importing KaggleHub; record one audio copy."""
    root = Path(__file__).resolve().parent
    # Cache under the project so repeated downloads can reuse files. setdefault
    # respects a cache path the user has explicitly configured elsewhere.
    os.environ.setdefault('KAGGLEHUB_CACHE', str(root / 'data' / 'kagglehub'))
    # KaggleHub reads environment settings on import, so the order matters.
    import kagglehub
    # This fetches the latest version; our completed experiment used version 1.
    # Pin /versions/1 in the handle for strict dataset-version reproduction.
    # Downloading provides data only; no model learns anything at this stage.
    path = kagglehub.dataset_download('uwrfkaggler/ravdess-emotional-speech-audio')
    # This Kaggle release contains two copies of the same 1,440 recordings.
    # Select the complete nested copy; never train on both copies.
    nested = Path(path) / 'audio_speech_actors_01-24'
    if nested.is_dir():
        path = str(nested)
    (root / 'data').mkdir(exist_ok=True)
    # Save the selected path so training need not guess Kaggle's folder layout.
    (root / 'data' / 'dataset_path.txt').write_text(str(path), encoding='utf-8')
    print(f'Dataset: {path}')


if __name__ == '__main__':
    main()
