"""Run dataset analysis, four baseline experiments, and optional frozen WavLM.

Run from the project directory: subprocess script paths and the saved dataset
path file are relative to the current working directory. The runner orchestrates
ravdess.py and finalize.py; it does not implement their learning algorithms.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(*arguments):
    """Execute one step in this Python environment and propagate failures."""
    # Use the same Python environment as this runner; check=True stops the
    # pipeline when any step fails rather than producing misleading later output.
    subprocess.run([sys.executable, *map(str, arguments)], check=True)


def main():
    """Construct the experiment list, train sequentially, and finalize once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Use an already extracted dataset instead of downloading')
    parser.add_argument('--output', default='runs/reproduction', help='Must be a new directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'])
    parser.add_argument('--include-wavlm', action='store_true', help='Add a frozen WavLM candidate to the existing four experiments')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if args.data:
        data = args.data
    else:
        run('download_data.py')
        data = Path('data/dataset_path.txt').read_text(encoding='utf-8').strip()
    # Data-health inspection precedes training. It produces reports, not weights.
    run('analyze.py', '--data', data, '--output', output / 'analysis')
    # Each tuple is (name, model, weight seed, frequency regions, max epochs,
    # early-stopping patience, batch size). These are the four experiments actually
    # run, not an exhaustive architecture or hyperparameter search.
    specs = [
        ('cnn_seed42', 'cnn', 42, 1, 100, 20, 16),
        ('cnn_bands_seed42', 'cnn', 42, 4, 150, 30, 16),
        ('mfcc_seed42', 'mfcc', 42, 4, 150, 25, 32),
        ('cnn_bands_seed43', 'cnn', 43, 4, 150, 30, 16),
    ]
    if args.include_wavlm:
        # pool-bands is passed uniformly but only affects CNN construction.
        specs.append(('wavlm_seed42', 'wavlm', 42, 4, 100, 20, 32))
    paths = []
    for name, model, seed, bands, epochs, patience, batch_size in specs:
        # Run sequentially to fit comfortably on the 4 GB GPU. All candidates
        # retain split seed 42; varying weight seed does not change test actors.
        target = output / name
        run('ravdess.py', 'train', '--data', data, '--device', args.device, '--output', target,
            '--model', model, '--seed', seed, '--pool-bands', bands, '--epochs', epochs,
            '--patience', patience, '--batch-size', batch_size, '--skip-test')
        paths.append(target)
    # Only now use test actors, after selecting the best validation checkpoint.
    run('finalize.py', '--runs', *paths, '--output', output / 'final', '--device', args.device)


if __name__ == '__main__':
    main()
