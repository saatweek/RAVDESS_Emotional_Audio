"""Standalone, offline Plotly reports for dataset exploration and model results."""
import html
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def write_report(path, title, introduction, figures):
    # Reporting does not train or change weights. Embed Plotly once per document
    # so charts work offline without copying the library for every figure.
    # Static images are easier to paste into slides, but lose hover/zoom details.
    sections = []
    for index, (heading, description, figure) in enumerate(figures):
        figure.update_layout(template='plotly_white', font=dict(family='Arial', size=14),
                             margin=dict(l=70, r=30, t=65, b=75), height=figure.layout.height or 520)
        chart = figure.to_html(full_html=False, include_plotlyjs=index == 0,
                               config=dict(responsive=True, displaylogo=False), div_id=f'chart-{index}')
        # Escape ordinary text so filenames/descriptions are not treated as HTML.
        sections.append(f'<section><h2>{html.escape(heading)}</h2><p>{html.escape(description)}</p>{chart}</section>')
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>''' + html.escape(title) + '''</title><style>
body{margin:0;background:#edf2f7;color:#183047;font-family:Arial,sans-serif}
main{max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:36px}
p{line-height:1.65;max-width:1000px}section{background:white;border-radius:14px;padding:24px;margin:24px 0;box-shadow:0 3px 14px #1830470a}
h2{margin-top:0;font-size:23px}.intro{font-size:17px}footer{padding:20px 0;color:#526477}
</style></head><body><main><h1>''' + html.escape(title) + '</h1><p class="intro">' + html.escape(introduction) + '</p>' + ''.join(sections) + '<footer>RAVDESS • PyTorch • Plotly | Charts work offline. Drag to zoom; double-click to reset.</footer></main></body></html>'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding='utf-8')


def training_report(path, history, metrics, split):
    # Learning curves show changes across epochs. Falling training loss with
    # worsening validation loss is a possible overfitting signal; it is not proof
    # of the cause. Also inspect the validation metric used for model selection.
    figures = []
    if history:
        curve = make_subplots(rows=1, cols=2, subplot_titles=['Loss', 'Accuracy and macro F1'])
        epochs = [h['epoch'] for h in history]
        for key, name in [('train_loss', 'Training loss (class weighted)'), ('validation_loss', 'Validation loss (unweighted)')]:
            curve.add_trace(go.Scatter(x=epochs, y=[h[key] for h in history], name=name), row=1, col=1)
        for key, name in [('train_accuracy', 'Training accuracy'), ('validation_accuracy', 'Validation accuracy'), ('validation_macro_f1', 'Validation macro F1')]:
            curve.add_trace(go.Scatter(x=epochs, y=[h[key] for h in history], name=name), row=1, col=2)
        curve.update_xaxes(title_text='Epoch')
        curve.update_yaxes(range=[0, 1], row=1, col=2)
        curve.update_layout(legend=dict(orientation='h', y=-0.22))
        figures.append(('Learning curves', 'Training includes augmentation and dropout; validation does not. Losses use different weighting and are not directly comparable.', curve))
    cm = np.array(metrics['confusion_matrix'])
    # Row-normalize: each cell becomes a fraction of that TRUE emotion. Its
    # diagonal equals per-class recall. Column normalization would instead answer
    # how pure each predicted class is, relating to precision.
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for matrix, label, fmt in [(cm, 'Recording counts', 'd'), (normalized, 'Recall within each true emotion', '.1%')]:
        fig = go.Figure(go.Heatmap(z=matrix.tolist(), x=EMOTIONS, y=EMOTIONS, colorscale='Blues',
                                   text=matrix.tolist(), texttemplate='%{text:' + fmt + '}',
                                   hovertemplate='True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>'))
        fig.update_layout(xaxis_title='Predicted emotion', yaxis_title='True emotion', yaxis_autorange='reversed')
        figures.append((f'{split} confusion matrix — {label}', 'The diagonal contains correct predictions. Rows identify the recorded emotion.', fig))
    fig = go.Figure()
    for key in ['precision', 'recall', 'f1-score']:
        fig.add_bar(x=EMOTIONS, y=[metrics['report'][e][key] for e in EMOTIONS], name=key)
    fig.update_layout(barmode='group', yaxis=dict(title='Score', range=[0, 1]))
    figures.append(('Per-emotion performance', 'Macro F1 gives each emotion equal weight despite the smaller neutral class.', fig))
    probs = np.array(metrics['probabilities'])
    truth = np.array(metrics['truth'])
    pred = np.array(metrics['predictions'])
    # "confidence" is only a convenient variable name for the largest softmax
    # score. This histogram is not a calibration curve or uncertainty interval.
    confidence = probs.max(axis=1)
    fig = go.Figure()
    for mask, name in [(truth == pred, 'Correct'), (truth != pred, 'Incorrect')]:
        fig.add_trace(go.Histogram(x=confidence[mask].tolist(), name=name, xbins=dict(start=0, end=1, size=.05), opacity=.65))
    fig.update_layout(barmode='overlay', xaxis_title='Largest softmax score', yaxis_title='Recordings')
    figures.append(('Prediction scores', 'Softmax scores are not calibrated confidence estimates. High-scoring mistakes can still occur.', fig))
    write_report(path, f'RAVDESS — {split.lower()} results',
                 f'{split} accuracy: {metrics["accuracy"]:.1%}. Macro F1: {metrics["macro_f1"]:.3f}. '
                 f'{len(truth)} recordings. Actor-exclusive evaluation; eight speech emotions.', figures)


def prediction_report(path, audio, result, model_type='cnn'):
    fig = go.Figure(go.Bar(x=list(result['probabilities']), y=list(result['probabilities'].values()), marker_color='#267bba'))
    fig.update_layout(yaxis=dict(title='Softmax score', range=[0, 1]), xaxis_title='Emotion')
    description = ('a centered four-second segment after silence trimming' if model_type == 'cnn'
                   else 'MFCC statistics across the recording after silence trimming')
    write_report(path, f'Predicted emotion: {result["prediction"]}',
                 f'Audio file: {Path(audio).name}. The model uses {description}.',
                 [('Emotion scores', 'These scores are not a validated assessment of a person’s internal emotions.', fig)])
