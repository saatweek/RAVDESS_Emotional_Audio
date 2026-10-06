"""Download a published emotion model into a new folder and verify its files.

Run with --repo saatweek/wavlm-ravdess-emotion. Use --revision with a Hub commit
to pin a release; the default main reference resolves to a commit before download.
The downloaded checkpoint uses ravdess.load_checkpoint, not remote Python code.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download


def verify_release(folder):
    """Check declared file sizes/hashes and require the complete local encoder."""
    folder = Path(folder)
    manifest = json.loads((folder / 'release_manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    required = {'best.pt', 'wavlm_encoder/config.json',
                'wavlm_encoder/preprocessor_config.json', 'wavlm_encoder/model.safetensors'}
    if not required.issubset(files):
        raise ValueError('The release does not describe a complete WavLM checkpoint.')
    for name, expected in files.items():
        # Treat manifest names as relative file paths, never outside destinations.
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError(f'Invalid release path: {name}')
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing or incomplete model file: {name}')
        digest = hashlib.sha256()
        # Stream the large encoder file rather than reading it all into RAM.
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Model file hash mismatch: {name}')
    return manifest


def main():
    """Resolve one Hub revision, download its package, and report verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Hugging Face owner/model name')
    parser.add_argument('--revision', default='main', help='Branch, tag, or immutable Hub commit')
    parser.add_argument('--output', default='runs/downloaded_wavlm', help='Must be a new directory')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    # Pin the resolved commit before fetching files so a moving branch cannot mix
    # checkpoint and encoder versions during this download. No login is needed
    # for a public model; the Hub library can use a saved login for private ones.
    revision = HfApi().model_info(args.repo, revision=args.revision).sha
    output.mkdir(parents=True, exist_ok=False)
    snapshot_download(repo_id=args.repo, revision=revision, local_dir=output,
                      allow_patterns=['best.pt', 'wavlm_encoder/*', 'release_manifest.json',
                                      'evaluation.json', 'README.md', 'NOTICE.md'])
    # Integrity checking detects incomplete/corrupted files; trust in a publisher
    # is a separate question. This helper imports no code from the downloaded repo.
    verify_release(output)
    print(f'Verified model: https://huggingface.co/{args.repo}/tree/{revision}')
    print(f'Checkpoint: {(output / "best.pt").resolve()}')


if __name__ == '__main__':
    main()
