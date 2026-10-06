---
language:
- en
library_name: pytorch
pipeline_tag: audio-classification
base_model: microsoft/wavlm-base-plus
tags:
- wavlm
- speech-emotion-recognition
- ravdess
- frozen-encoder
- custom-classifier
model-index:
- name: Frozen WavLM + RAVDESS emotion classifier
  results:
  - task:
      type: audio-classification
      name: Speech emotion classification
    dataset:
      name: RAVDESS speech, four held-out actors
      type: ravdess
    metrics:
    - type: accuracy
      name: Accuracy
      value: 0.7166666666666667
    - type: f1
      name: Macro F1
      value: 0.7038388576974692
---

# Frozen WavLM + RAVDESS emotion classifier

An eight-class speech emotion classifier using Microsoft's frozen
`microsoft/wavlm-base-plus` encoder and a small trained PyTorch classification
head. **The encoder was not fine-tuned.** Only the emotion head was trained on
RAVDESS speech.

Maintainer: [saatweek](https://huggingface.co/saatweek).
Source, full explanations, tests and Gradio app:
[GitHub project](https://github.com/saatweek/RAVDESS_Emotional_Audio).
Model repository:
[saatweek/wavlm-ravdess-emotion](https://huggingface.co/saatweek/wavlm-ravdess-emotion).

## Labels and architecture

Output order: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Audio is downmixed to mono, resampled to 16 kHz, edge-trimmed at a relative 35 dB
threshold, then center-cropped to at most four seconds. Short clips are not
zero-padded; fewer than 400 samples after preprocessing are rejected.

The encoder's final hidden states have 768 channels. Concatenating time-wise
mean and population standard deviation produces 1,536 features. Feature-wise
normalization is fitted only on training actors and saved as head buffers.

Head: 1,536 → 256 linear, LayerNorm, ReLU, dropout 0.35 → eight logits.
Softmax is applied when reporting scores; those scores are not calibrated
confidence or a measurement of a person's internal emotional state.

Base encoder revision:
`4c66d4806a428f2e922ccfa1a962776e232d487b`.
The local encoder and processor are included so the project's existing loader
can perform inference without downloading a different upstream revision.

## Evaluation

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Original selected CNN | 0.4834 | 55.42% | 0.5324 |
| Frozen WavLM + trained head | 0.7042 | **71.67% (172/240)** | **0.7038** |

Training: 960 recordings from actors
01, 02, 03, 04, 05, 07, 08, 09, 10, 13, 15, 17, 20, 21, 22, 23.
Validation: 240 recordings from actors 11, 12, 14, 18.
Test: 240 recordings from actors 06, 16, 19, 24.

Actor split seed: 42. Weight seed: 42. Both statements appear across partitions.
Best validation checkpoint: epoch 4; training stopped at epoch 24 with patience
20. Head batch size 32, AdamW initial LR 0.001, weight decay 0.01, weighted
cross entropy, gradient clipping and validation-driven learning-rate reduction.

Checkpoint/candidate selection used validation macro F1. The later candidate
comparison included WavLM alongside four CNN/MFCC candidates on identical
manifests. The original CNN test result had already been inspected before that
later experiment. Future tuning needs a fresh evaluation protocol.

Test accuracy by actor: 06 = 70%, 16 = 71.67%, 19 = 80%, 24 = 65%.
The comparison changes both representation and head architecture; it does not
isolate the effect of pretraining alone. Exact class scores/confusion matrices
are in `evaluation.json` and the
[written WavLM results](https://github.com/saatweek/RAVDESS_Emotional_Audio/blob/main/WAVLM_RESULTS.md).

## Download and predict

This is a **custom PyTorch head**, not a standard
`AutoModelForAudioClassification` export. Use the project loader below; the
Hub task tag describes its purpose and does not promise a hosted inference widget.
No downloaded Python code is executed through trust_remote_code.

Clone the GitHub repository, create its Python environment and install its
dependencies as described in the project README. From that directory:

```powershell
.\.venv\Scripts\python.exe download_model.py --repo saatweek/wavlm-ravdess-emotion --output runs/downloaded_wavlm
.\.venv\Scripts\python.exe ravdess.py predict --checkpoint runs/downloaded_wavlm/best.pt --audio sample_audio.wav --device auto
```

The downloader creates a new output directory, verifies the release file hashes,
and prints the resolved Hub commit. Use `--revision COMMIT_HASH` for a pinned
download. To use the downloaded model in Gradio, set the WavLM entry in
MODEL_PATHS in app.py to `ROOT / 'runs/downloaded_wavlm/best.pt'`.
The other two app options still need their own trained CNN/MFCC checkpoints.

The equivalent library flow, after downloading, is:

```python
import torch
from ravdess import EMOTIONS, load_checkpoint

checkpoint, model, extract = load_checkpoint("runs/downloaded_wavlm/best.pt", "cpu")
x = torch.from_numpy(extract("sample_audio.wav", checkpoint["config"]))[None]
with torch.inference_mode():
    scores = model(x).softmax(1)[0].tolist()
print(dict(zip(EMOTIONS, scores)))
```

## Repository files

- `best.pt`: head state dictionary including train-fitted buffers, labels,
  preprocessing metadata and selected epoch.
- `wavlm_encoder/`: unchanged pretrained encoder safetensors, architecture
  config and audio processor settings.
- `evaluation.json`: aggregate validation/test metrics, per-class report,
  confusion matrix and per-actor test metrics; no recording paths.
- `release_manifest.json`: file sizes/hashes, preprocessing, training summary
  and export source commit.
- `README.md`: this model card.
- `NOTICE.md`: upstream model attribution and dataset licensing information.

No dataset recordings, local split manifests, training caches, personal filesystem
paths or credentials are included.

## Intended use and limitations

Intended for learning, reproducible experiments and portfolio demonstrations of
acted English speech emotion classification. RAVDESS is a small dataset with two
statements and professional actors. A single held-out actor split does not
establish reliability on spontaneous speech, arbitrary accents, languages,
microphones or background noise. Neutral/happy/sad remain difficult classes.

The model independently classifies a segment. The Gradio app repeats inference
on rolling windows; the model itself has no conversational memory, transcription,
temporal smoothing or online learning. Do not present its scores as a validated
assessment of mental state.

## Attribution and licensing information

Microsoft WavLM authors: Sanyuan Chen and collaborators.
[Model card](https://huggingface.co/microsoft/wavlm-base-plus) ·
[Paper](https://arxiv.org/abs/2110.13900).

RAVDESS: Livingstone SR, Russo FA (2018), PLoS ONE 13(5): e0196391.
[Paper](https://doi.org/10.1371/journal.pone.0196391) ·
[Dataset and CC BY-NC-SA 4.0 information](https://zenodo.org/records/1188976).

See `NOTICE.md` for component-specific notices. This release does not assert
a blanket MIT license or relicense Microsoft's encoder. No separate license
for the project's trained head has been specified by the maintainer.
