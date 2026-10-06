"""Select by validation macro F1, then evaluate the chosen checkpoint on test actors.

Works with CNN, MFCC, and WavLM runs through the shared checkpoint factory.
Study this after ravdess.train/score; it assembles an evaluation, not a new model.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import f1_score

from ravdess import device_for, loader, load_checkpoint, score, save_json, plot_results, EMOTIONS
from reporting import write_report


def main():
    """Require matching manifests, copy the winner, and export test evidence."""
    # Model selection is itself a learning decision. Comparing several models on
    # the TEST set and reporting the winner would bias the claimed final result.
    # Read each saved VALIDATION score first, freeze the winner, then test it.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', default='runs/final')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    candidates, reference = [], None
    for name in args.runs:
        root = Path(name)
        splits = json.loads((root / 'splits.json').read_text())
        # A score on easier actors is not a fair comparison with a score on harder
        # actors. Require identical manifests, including recording order.
        if reference is not None and splits != reference:
            raise ValueError('All candidates must use identical train, validation and test recordings.')
        reference = splits
        metrics = json.loads((root / 'validation_metrics.json').read_text())
        candidates.append(dict(run=str(root), validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1']))
    # max keeps the first candidate on an exact tie; no test-based tie-breaking.
    winner = max(candidates, key=lambda r: r['validation_macro_f1'])
    # Persist the selection before any test predictions are made.
    save_json(output / 'selection.json', dict(criterion='validation_macro_f1', candidates=candidates, selected=winner))
    source = Path(winner['run'])
    # Copy the chosen checkpoint; we do not retrain it on validation/test audio.
    # Retraining on train+validation is possible, but requires a predefined final
    # training schedule and would produce a different checkpoint to evaluate.
    for name in ['best.pt', 'splits.json', 'config.json', 'history.json', 'validation_metrics.json']:
        shutil.copy2(source / name, output / name)
    if (source / 'wavlm_encoder').is_dir():
        # best.pt contains only WavLM head weights; inference also needs encoder.
        shutil.copytree(source / 'wavlm_encoder', output / 'wavlm_encoder')
    device = device_for(args.device)
    checkpoint, model, extract = load_checkpoint(output / 'best.pt', device)
    rows = reference['test']
    metrics = score(model, loader(rows, checkpoint['config'], 16, cache='data/features', extract=extract), device)
    # loader/score retain manifest order, so row IDs align with these predictions.
    actors = np.array([r['actor'] for r in rows])
    truth, predictions = np.array(metrics['truth']), np.array(metrics['predictions'])
    per_actor = []
    # Break out performance by actor: an overall average can hide a model that
    # works well for some voices but poorly for others. Four test actors are a
    # small sample; this is descriptive analysis, not a population guarantee.
    for actor in sorted(set(actors)):
        mask = actors == actor
        per_actor.append(dict(actor=int(actor), recordings=int(mask.sum()), accuracy=float(np.mean(truth[mask] == predictions[mask])),
                              macro_f1=f1_score(truth[mask], predictions[mask], labels=list(range(8)), average='macro', zero_division=0)))
    metrics['per_actor'] = per_actor
    # A simple reference: always choosing one of the largest classes gives 13.3%
    # on this fixed split. This is a class-count calculation, not a trained model.
    metrics['majority_baseline_accuracy'] = float(max(np.bincount(truth)) / len(truth))
    save_json(output / 'test_metrics.json', metrics)
    # Preserve individual errors and all eight scores so the report can be
    # checked against concrete recordings instead of only aggregate metrics.
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file', 'actor', 'true_emotion', 'predicted_emotion', 'correct', *[f'prob_{e}' for e in EMOTIONS]])
        for row, prediction, probabilities in zip(rows, predictions, metrics['probabilities']):
            writer.writerow([row['path'], row['actor'], EMOTIONS[row['label']], EMOTIONS[prediction], row['label'] == prediction, *probabilities])
    # Show training/validation history beside the restored winner's test scores.
    # Candidate bars use validation; actor bars use only the selected model's test.
    history = json.loads((output / 'history.json').read_text())
    plot_results(output, history, metrics)
    fig = go.Figure()
    for metric in ['validation_accuracy', 'validation_macro_f1']:
        fig.add_bar(x=[Path(r['run']).name for r in candidates], y=[r[metric] for r in candidates], name=metric.replace('_', ' '))
    fig.update_layout(barmode='group', yaxis=dict(range=[0, 1], title='Validation score'))
    actor_plot = go.Figure()
    for metric in ['accuracy', 'macro_f1']:
        actor_plot.add_bar(x=[f'Actor {r["actor"]:02d}' for r in per_actor], y=[r[metric] for r in per_actor], name=metric)
    actor_plot.update_layout(barmode='group', yaxis=dict(range=[0, 1], title='Test score'))
    write_report(output / 'comparison.html', 'Model selection and actor performance',
                 f'Selected {source.name} using validation macro F1 only. The selected model was then evaluated on four held-out actors. '
                 'A single actor split gives a limited estimate of generalization; performance can differ for other voices and recording conditions.',
                 [('Candidate comparison', 'Every candidate uses the same actor split. No test metric is used to choose the winner.', fig),
                  ('Held-out actor performance', 'Each actor contributes 60 test recordings. Variation across actors matters when interpreting the overall result.', actor_plot)])
    print(json.dumps(dict(selected=winner, test_accuracy=metrics['accuracy'], test_macro_f1=metrics['macro_f1'], per_actor=per_actor), indent=2))


if __name__ == '__main__':
    main()
