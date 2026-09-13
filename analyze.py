"""Audit RAVDESS audio and produce an interactive Plotly exploration report."""
import argparse
import csv
import hashlib
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
from plotly.subplots import make_subplots

from ravdess import EMOTIONS, CONFIG, records, save_json, split_records
from reporting import write_report


def main():
    # Start with data quality, before interpreting model accuracy. A model can
    # learn from wrong labels, duplicated files or corrupt inputs without making
    # the underlying problem obvious in its final accuracy number.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='runs/analysis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = records(args.data)
    splits = split_records(rows, args.seed)
    # Metadata checks can cover all files. Feature examples below are restricted
    # to training actors; we do not choose model settings from test predictions.
    assignment = {r['actor']: key for key, values in splits.items() for r in values}
    hashes, audit, duplicates = {}, [], []
    for index, row in enumerate(rows):
        path = Path(row['path'])
        # Inspect original audio, without resampling or trimming. always_2d gives
        # [samples, channels] even for mono, so channel counts remain inspectable.
        audio, sr = sf.read(path, always_2d=True)
        if not np.isfinite(audio).all() or not len(audio):
            raise ValueError(f'Invalid audio: {path}')
        # A hash is a fingerprint of file BYTES. It catches exact copies even
        # with different names, but not re-encoded copies or near-duplicate sound.
        # Perceptual/audio-sample matching would be a stronger additional audit.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            previous = hashes[digest]
            if assignment[previous['actor']] != assignment[row['actor']]:
                raise ValueError(f'Identical audio crosses splits: {path} and {previous["path"]}')
            # Retain the original named trials when the duplicate is within a
            # split, and record it transparently. Deduplication was an alternative;
            # here the one pair is in training and cannot directly inflate test.
            duplicates.append(dict(first=previous['path'], second=str(path), split=assignment[row['actor']]))
        hashes[digest] = row
        parts = list(map(int, path.stem.split('-')))
        # Duration = samples/sample_rate. RMS = sqrt(mean(amplitude squared)),
        # a signal-level measure, not perceived loudness or a direct emotion label.
        # Near-full-scale samples flag possible clipping, not proof of distortion.
        audit.append(dict(**row, emotion=EMOTIONS[row['label']], split=assignment[row['actor']],
                          intensity=parts[3], statement=parts[4], repetition=parts[5],
                          duration=len(audio)/sr, sample_rate=sr, channels=audio.shape[1],
                          peak=float(np.max(np.abs(audio))), rms=float(np.sqrt(np.mean(audio**2))),
                          clipping_fraction=float(np.mean(np.abs(audio) >= 0.999)), sha256=digest))
        if (index + 1) % 240 == 0:
            print(f'Audited {index + 1}/{len(rows)} recordings', flush=True)
    # Keep a row per file so someone can audit the summary rather than relying
    # only on plots. This CSV is metadata; it is not the CNN's training input.
    with (output / 'audio_inventory.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = dict(recordings=len(rows), actors=len(assignment), total_minutes=sum(r['duration'] for r in audit)/60,
                   sample_rates=sorted({r['sample_rate'] for r in audit}), channels=sorted({r['channels'] for r in audit}),
                   min_duration=min(r['duration'] for r in audit), max_duration=max(r['duration'] for r in audit),
                   duplicate_hashes=len(duplicates), duplicate_pairs=duplicates, cross_split_duplicates=0,
                   all_finite=True, silent_files=sum(r['peak'] == 0 for r in audit),
                   files_with_clipping=sum(r['clipping_fraction'] > 0 for r in audit),
                   class_counts={e: sum(r['emotion'] == e for r in audit) for e in EMOTIONS},
                   split_actors={key: sorted({r['actor'] for r in values}) for key, values in splits.items()})
    save_json(output / 'audit.json', summary)
    save_json(output / 'splits.json', splits)
    # Read the charts as questions: are classes balanced, are actors represented,
    # do lengths vary, and are there suspicious amplitude/duration outliers?
    # These describe associations, not causal evidence about emotion.
    figures = []
    fig = go.Figure()
    for split in ['train', 'validation', 'test']:
        fig.add_bar(x=EMOTIONS, y=[sum(r['emotion'] == e and r['split'] == split for r in audit) for e in EMOTIONS], name=split)
    fig.update_layout(barmode='stack', yaxis_title='Recordings')
    figures.append(('Emotion balance by split', 'Neutral has one intensity level; the other emotions have two. Split assignment depends only on actor IDs.', fig))
    actors = sorted(assignment)
    matrix = [[sum(r['actor'] == a and r['emotion'] == e for r in audit) for a in actors] for e in EMOTIONS]
    fig = go.Figure(go.Heatmap(z=matrix, x=[f'{a:02d} ({assignment[a]})' for a in actors], y=EMOTIONS,
                              colorscale='Blues', text=matrix, texttemplate='%{text}'))
    fig.update_layout(xaxis_title='Actor and assigned split', yaxis_title='Emotion')
    figures.append(('Actor coverage', 'Each actor belongs to exactly one split. This prevents recordings from the same voice appearing in both training and testing.', fig))
    fig = go.Figure()
    for emotion in EMOTIONS:
        fig.add_trace(go.Box(y=[r['duration'] for r in audit if r['emotion'] == emotion], name=emotion, boxpoints='outliers'))
    fig.update_layout(yaxis_title='Original duration (seconds)', showlegend=False)
    figures.append(('Recording lengths', 'These are original file durations, before trimming and the model’s four-second crop or padding.', fig))
    fig = go.Figure()
    for emotion in EMOTIONS:
        subset = [r for r in audit if r['emotion'] == emotion]
        fig.add_trace(go.Scatter(x=[r['duration'] for r in subset], y=[r['rms'] for r in subset],
                                mode='markers', name=emotion, text=[Path(r['path']).name for r in subset],
                                hovertemplate='%{text}<br>Duration: %{x:.2f}s<br>RMS: %{y:.4f}<extra></extra>'))
    fig.update_layout(xaxis_title='Duration (seconds)', yaxis_title='RMS amplitude')
    figures.append(('Duration and amplitude', 'Amplitude differences can reflect the recording or performance and do not establish emotion on their own.', fig))
    # Feature examples come only from training actors. Show original duration
    # here for interpretation; feature() crops/pads the actual CNN inputs.
    # 160 samples / 16,000 samples per second = 0.01 seconds between columns.
    fig = make_subplots(rows=4, cols=2, subplot_titles=EMOTIONS, vertical_spacing=.08)
    for index, emotion in enumerate(EMOTIONS):
        row = next(r for r in audit if r['emotion'] == emotion and r['split'] == 'train')
        audio, sr = librosa.load(row['path'], sr=CONFIG['sr'])
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        fig.add_trace(go.Heatmap(z=mel.tolist(), x=(np.arange(mel.shape[1])*.01).tolist(),
                                 colorscale='Viridis', showscale=False, zmin=-80, zmax=0), row=index//2+1, col=index%2+1)
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='Mel band')
    fig.update_layout(height=1150)
    figures.append(('Log-mel examples from training actors', 'One original-duration example per emotion. Colors show decibels relative to the peak of each recording.', fig))
    report = output / 'dataset_report.html'
    write_report(report, 'RAVDESS — dataset analysis',
                 f'{len(rows):,} recordings · {len(assignment)} actors · {summary["total_minutes"]:.1f} minutes. '
                 f'Validated audio decoding and finite samples. Found {len(duplicates)} byte-identical recording pairs, all within the same split. '
                 f'{summary["files_with_clipping"]} files contain samples at or above 0.999 amplitude. '
                 'Metadata and file-health checks cover all files; illustrative features use training actors.', figures)
    print(f'Report: {report.resolve()}')


if __name__ == '__main__':
    main()
