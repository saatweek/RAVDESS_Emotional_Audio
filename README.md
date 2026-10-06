# RAVDESS Speech Emotion Classification

An end-to-end PyTorch project that classifies eight acted speech emotions:
neutral, calm, happy, sad, angry, fearful, disgust and surprised. It includes
dataset download/audit, CNN and MFCC baselines, a frozen pretrained WavLM encoder
with a trained emotion head, actor-exclusive evaluation, offline Plotly reports,
and a Gradio app with uploads and continuous microphone recording.

**Current best reported pipeline: WavLM**, with **71.67% test accuracy
(172/240)** and **0.7038 macro F1** on four held-out actors. The original selected
CNN achieved **55.42% (133/240)** and **0.5324 macro F1** on those same actors.
Candidate selection uses validation macro F1. These are results on one acted
speech split, not established accuracy on ordinary conversations.
See [WAVLM_RESULTS.md](WAVLM_RESULTS.md) and the historical [RESULTS.md](RESULTS.md).

## Get the project from GitHub

Repository: [saatweek/RAVDESS_Emotional_Audio](https://github.com/saatweek/RAVDESS_Emotional_Audio).

```powershell
git clone https://github.com/saatweek/RAVDESS_Emotional_Audio.git
cd RAVDESS_Emotional_Audio
```

The repository includes all project Python source, tests, dependency files,
study guides, written results and the existing sample recording. Dataset downloads,
feature caches, raw operational run folders and trained weights are
excluded from GitHub. Portable reports and metadata are included in
[runs/published](runs/published/README.md). The Python environment and Gradio's local sharing certificate
are also excluded. The trained WavLM bundle is available separately on
[Hugging Face](https://huggingface.co/saatweek/wavlm-ravdess-emotion), including
its frozen encoder; it can be downloaded instead of retrained. CNN and MFCC still
need their local trained checkpoints before the three-model web app can start.

Follow [installation](#install-on-another-machine), then
[dataset download](#download-and-inspect-data). To create the three model files
at the exact paths expected by app.py on a fresh clone, run:

```powershell
$datasetPath = (Get-Content data/dataset_path.txt -Raw).Trim()
.\.venv\Scripts\python.exe ravdess.py train --data $datasetPath --model cnn --pool-bands 4 --device auto --epochs 150 --patience 30 --skip-test --output runs/cnn_bands_seed42
.\.venv\Scripts\python.exe ravdess.py train --data $datasetPath --model mfcc --device auto --batch-size 32 --epochs 150 --patience 25 --skip-test --output runs/mfcc_seed42
.\.venv\Scripts\python.exe download_model.py --repo saatweek/wavlm-ravdess-emotion --output runs/wavlm_comparison
.\.venv\Scripts\python.exe finalize.py --runs runs/cnn_bands_seed42 --output runs/final --device auto
.\.venv\Scripts\python.exe -u app.py --share
```

These commands train CNN/MFCC, finalize CNN and download the published WavLM
release to populate the demo's fixed checkpoint paths. They are not the original multi-candidate
comparison; use [the full runner](#train-select-and-evaluate) to reproduce that
workflow. Training may produce different scores across environments. Each output
directory must be new; on the original laptop, reuse the already trained models
instead of running this fresh-clone recipe over existing folders.

To train WavLM yourself rather than download this release, follow
[the WavLM training section](#train-and-predict-with-wavlm).

The full runner writes models under its chosen output folder. To serve those
models instead, update MODEL_PATHS in app.py to their actual checkpoint locations.
Keep wavlm_encoder/ next to its WavLM best.pt. The public demo address is printed
at launch and changes on restart, so it is not hard-coded into this README.

## How to read and understand the project

Use these three documents together:

- **This README:** setup, commands, file responsibilities and saved artifacts.
- **[INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md):** concepts, decisions, results,
  limitations and practice questions.
- **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md):** statement-by-statement study
  notes, inputs/outputs, tensor shapes, call order and why each operation exists.

Read source in this order, rather than alphabetically:

| Order | File / section | What you should be able to explain afterward |
|---:|---|---|
| 1 | [Fourier_Transformation.py](Fourier_Transformation.py) | Samples, sine waves, time axes and FFT peaks |
| 2 | [Audio_prep.py](Audio_prep.py) | Waveform, whole-clip FFT, STFT and MFCC plots |
| 3 | [download_data.py](download_data.py) | Download location, cache defaults, choosing one dataset copy |
| 4 | [ravdess.py](ravdess.py): constants, `records`, `split_records` | Filename labels and speaker-exclusive splits |
| 5 | [analyze.py](analyze.py) | Original-file audit, duplicate hashes and training-only illustrations |
| 6 | [ravdess.py](ravdess.py): `feature`, `EmotionCNN`, `EmotionMLP` | CNN/MFCC preprocessing and learned layers |
| 7 | [wavlm.py](wavlm.py) | Raw-waveform encoding, freezing, pooling and the emotion head |
| 8 | [ravdess.py](ravdess.py): factory, loader, scoring, training, CLI | Caches, batches, gradients, selection, restoration and inference |
| 9 | [finalize.py](finalize.py), [reporting.py](reporting.py) | Validation-based model choice, test evidence and offline charts |
| 10 | [run_project.py](run_project.py) | How the experiment steps are orchestrated |
| 11 | [app.py](app.py) | Decode uploads, keep session windows, reuse checkpoints, serve requests |
| 12 | [test_ravdess.py](test_ravdess.py), [test_wavlm.py](test_wavlm.py), [test_app.py](test_app.py) | What is verified and what needs real data/browser checks |
| 13 | [download_model.py](download_model.py), [MODEL_CARD.md](MODEL_CARD.md) | Hub revision pinning, release integrity and published model provenance |

The walkthrough covers every project-owned Python file. Comments explain
purpose and choices near the source; use the walkthrough to trace a statement's
shape and effect. Generated reports, model tensors, third-party library internals
and audio samples are artifacts/dependencies, not additional source files to
memorize. Library calls should be explained by their contract before studying
their internal implementation.

## The three model paths

All paths produce eight **logits**; softmax is applied for display/evaluation.
The model receives acoustic features, never the emotion encoded in a filename.

| Model | Input preparation | Trainable network | Saved preprocessing |
|---|---|---|---|
| CNN | Mono 16 kHz, edge trim, centered 4-second crop or right zero padding; 64 log-mel bands and per-clip z-normalization | Four convolution blocks; selected variant retains four frequency regions | Audio settings and frequency pooling count |
| MFCC + NN | Full trimmed recording; 40 MFCC means/stds plus delta means/stds = 160 features | 160 → 256 → 128 → 8 MLP | Training-only feature means/stds stored as buffers |
| WavLM | Mono 16 kHz, edge trim, at most 4 centered seconds, no short-clip padding; frozen base-plus encoder; final-state mean/std = 1,536 features | 1,536 → 256 → 8 head; encoder weights stay frozen | Training-only buffers, encoder revision/settings and local encoder files |

CNN input is `[batch, 1, 64, 401]` with default settings. MFCC input is
`[batch, 160]`; WavLM head input is `[batch, 1536]`. WavLM runs on raw audio
internally, not on the CNN spectrogram or MFCC vectors.

All three trainable networks use class-weighted cross entropy, AdamW, gradient
clipping, validation-based learning-rate reduction and early stopping. CUDA
training uses mixed precision for the trainable network. The frozen WavLM
encoder extracts one clip at a time in float32; `--batch-size` controls head
training, not encoder extraction. Time/frequency masks apply only to CNN training.

## Gradio demo: start, check and share

The three trained checkpoints must exist before launch. Run from the project
folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -u app.py --share
```

Models load and warm up before the server prints its URLs. Open the printed
public HTTPS link on phones, tablets or computers. Keep the laptop awake, online
and the process running. Ctrl+C in that terminal stops the server. The share
link is temporary; Gradio's sharing guide documents its one-week lifetime:
[sharing guide](https://www.gradio.app/guides/sharing-your-app).
Restarting generally produces a new public URL; use the newly printed one.

For local use, omit `--share`. The default local URL is
[http://127.0.0.1:7860/](http://127.0.0.1:7860/). Add `--port 7861` to request
a different port or `--device cpu` to use CPU. This app explicitly requests a
port; an occupied port causes a launch error. Other local apps can use other
ports and addresses. **127.0.0.1 refers to the device opening the URL**; friends
should open the public share link.

To check the default port before starting another process:

```powershell
Test-NetConnection 127.0.0.1 -Port 7860
```

A successful port check shows a listener, which could be another application.
Open the URL and confirm it displays Speech Emotion Demo. If it does, reuse it;
if no listener is present, start the app. Check the selected port if you changed
`--port`. A local listener alone does not prove the public relay is reachable.

Choose CNN, MFCC + NN or WavLM. Upload accepts supported audio containers
including WAV, MP3, M4A, WebM, OGG, FLAC, AAC and AIFF, up to 30 seconds and
25 MB. Bundled FFmpeg decodes to mono 16 kHz; no separate FFmpeg install is
needed. CNN/WavLM use a centered segment of long uploads, while MFCC summarizes
the full trimmed upload. Command-line WAV prediction has no web 30-second cap.

On **Microphone — live**, allow microphone access and press record. The first
prediction needs four seconds of received audio; requests then update roughly
once per second while recording continues. Each visitor gets a separate bounded
four-second buffer. All models see that same window before their own saved
preprocessing. Very quiet windows show a status message. Press stop to finish.

This is repeated window inference. There is no conversational memory,
transcription, result smoothing or automatic model learning while you speak.
Network delay, decoding, queued work and inference add latency; the current
implementation does not discard stale pending chunks or guarantee an update
every wall-clock second. It has no prediction log or latency panel.

The app loads weights once from these paths (in [app.py](app.py)):

| Choice | Required checkpoint |
|---|---|
| CNN | `runs/final/best.pt` |
| MFCC + NN | `runs/mfcc_seed42/best.pt` |
| WavLM | `runs/wavlm_comparison/best.pt` and sibling `wavlm_encoder/` |

Inference is serialized with a shared lock. Upload and stream callbacks use
separate Gradio scheduler groups; that does not guarantee fair service under
load. A new recording resets its audio history; switching models clears scores
but keeps the window. Stopping preserves the last displayed prediction.

By default, Gradio stores uploads/recordings in `data/gradio_cache/` and checks
hourly for files older than an hour. The intermediate float WAV used for one
prediction is deleted when that call finishes. Environment defaults disable
analytics and choose the cache path; explicitly set environment variables take
precedence. Test the actual browser/microphone and public link before a demo.

## Install on another machine

Python 3.12 was used. For the verified Windows/NVIDIA environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The verified GPU is an RTX 3050 Ti Laptop GPU with 4 GB VRAM. The CUDA 12.4
wheel includes its runtime. Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for other platforms. `requirements.txt` specifies direct dependencies and tested
compatibility ranges; `requirements-lock.txt` records the installed environment.
Install the appropriate PyTorch wheel first if using the lock file.

## Download and inspect data

```powershell
.\.venv\Scripts\python.exe download_data.py
$datasetPath = (Get-Content data/dataset_path.txt -Raw).Trim()
.\.venv\Scripts\python.exe analyze.py --data $datasetPath
```

The KaggleHub downloader requests
`uwrfkaggler/ravdess-emotional-speech-audio`; the completed experiment used
version 1. The code currently requests the latest version, so strict reproduction
requires pinning the dataset version as its source comment explains. It defaults
to a project-local cache and selects the nested `audio_speech_actors_01-24`
folder when present, avoiding the duplicate folder copy in version 1.

The audit validates decoding/finite samples and records classes, actors, sample
rates, channels, duration, RMS/peak amplitude and file-byte SHA-256 hashes.
One byte-identical actor-07 pair stays in training; none crosses splits.
Metadata checks cover all files; illustrative log-mel examples use training actors.
Audit charts show original-length audio, not the exact normalized CNN input.

## Download the published WavLM model

Model: [saatweek/wavlm-ravdess-emotion](https://huggingface.co/saatweek/wavlm-ravdess-emotion).
Initial verified release: `d5ecc1a27ea641a2131b1a877d3600f4bd800a3a`.
The release includes the trained custom head, unchanged frozen encoder/processor,
model card, aggregate evaluation, provenance notices and file hashes. No dataset
recordings or local path manifests are uploaded.

After installing this project's dependencies, download into a new folder:

```powershell
.\.venv\Scripts\python.exe download_model.py --repo saatweek/wavlm-ravdess-emotion --output runs/downloaded_wavlm
.\.venv\Scripts\python.exe ravdess.py predict --checkpoint runs/downloaded_wavlm/best.pt --audio sample_audio.wav --device auto
```

The helper resolves main to a Hub commit before downloading and verifies file
sizes and SHA-256 hashes. It prints the resolved revision; use `--revision COMMIT`
to pin a release later. It refuses an existing output directory and does not
execute Python from the Hub repository. Loading uses this project's custom
`ravdess.load_checkpoint` implementation, not a standard Transformers classifier.

For the app, point its WavLM MODEL_PATHS entry at the downloaded best.pt, or on a
fresh clone download directly into runs/wavlm_comparison as shown above. CNN/MFCC
remain separate trained options. Model storage on the Hub does not host the app;
your laptop still runs Gradio. See [MODEL_CARD.md](MODEL_CARD.md) for full details.

## Train and predict with WavLM

To use the trained release immediately, follow the download section above.
The commands below train a new head.

Hugging Face Transformers loads
[`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus).
This pretrained encoder needs a trained emotion head to predict our labels.
The code freezes it and caches embeddings, avoiding repeated encoder passes
during head-training epochs and fitting the laptop's 4 GB GPU.

```powershell
$datasetPath = (Get-Content data/dataset_path.txt -Raw).Trim()
.\.venv\Scripts\python.exe ravdess.py train --data $datasetPath --model wavlm --device cuda --batch-size 32 --epochs 100 --patience 20 --skip-test --output runs/my_wavlm
.\.venv\Scripts\python.exe ravdess.py predict --checkpoint runs/my_wavlm/best.pt --audio sample_audio.wav --device cuda --output runs/my_wavlm_prediction
```

The first run downloads to `data/huggingface/`. A moving revision such as
`main` is resolved to an immutable Hub commit. Features in `data/features/`
are keyed by preprocessing/configuration plus file path, size and modification
time. This is a metadata cache key, not a file-content hash. WavLM settings also
include the encoder revision, pooling, feature version and Transformers version.
Rebuild CNN/MFCC caches after changing their extraction code; bump the WavLM
feature version when changing its embedding semantics.

Each WavLM run exports `best.pt` and `wavlm_encoder/`. Keep them together:
checkpoint loading restores the head and local encoder offline. New audio is
encoded again at prediction time; live predictions do not use a training cache.

## Train, select and evaluate

Every experiment output must be a new directory. For one CNN candidate:

```powershell
$datasetPath = (Get-Content data/dataset_path.txt -Raw).Trim()
.\.venv\Scripts\python.exe ravdess.py train --data $datasetPath --model cnn --pool-bands 4 --device cuda --epochs 150 --patience 30 --skip-test --output runs/new_cnn
```

Use `--model mfcc` for the alternative MLP. Defaults are batch size 16, seed 42
and split seed 42; batch size 32 was used for the MFCC/WavLM experiments.
Explicit `--device cuda` fails if unavailable; `auto` falls back to CPU.

Reserve test predictions with `--skip-test`, then choose among runs using
validation macro F1 and evaluate the fixed winner:

```powershell
.\.venv\Scripts\python.exe finalize.py --runs runs/cnn_bands_seed42 runs/my_wavlm --output runs/my_comparison --device cuda
```

Finalization requires identical manifests, saves its selection before inference,
copies the winning model, then exports test metrics/predictions. Use this only
with matching runs that exist. To re-score a saved checkpoint without retraining:

```powershell
.\.venv\Scripts\python.exe ravdess.py evaluate --checkpoint runs/wavlm_comparison/best.pt --manifest runs/wavlm_comparison/splits.json --device cuda --output runs/reevaluation
```

For the full four-run reproduction, with optional WavLM as a fifth candidate:

```powershell
.\.venv\Scripts\python.exe run_project.py --data $datasetPath --output runs/reproduction --device cuda --include-wavlm
```

Omit `--include-wavlm` for the original four experiments; omit `--data` to
download first. The runner audits, trains sequentially and finalizes; failed
steps stop it. This reproduces the workflow, not guaranteed bit-identical weights.
Checkpoints contain model state and metadata, not full optimizer/RNG resume state.

## Evaluation and saved artifacts

The completed experiment artifacts are available in
[runs/published/](runs/published/README.md): learning histories, configurations,
portable split manifests, validation/test metrics, prediction CSVs and interactive
reports. Start with the [WavLM report](runs/published/wavlm_comparison/report.html)
and [candidate comparison](runs/published/wavlm_comparison/comparison.html).

Clone or [download the repository ZIP](https://github.com/saatweek/RAVDESS_Emotional_Audio/archive/refs/heads/main.zip),
then open an HTML report in your browser. GitHub displays HTML source rather than
running the charts. Keep runs/published/_assets/ with the reports: one shared local
Plotly library makes them work offline while reducing duplicated file size.

These are historical snapshots. Local paths are replaced by DATASET_ROOT,
PROJECT_ROOT or HOME_ROOT placeholders; metric values and ordering are preserved.
Rebase published dataset paths before using a split manifest for evaluation.
The original local run folders remain operational and were not modified. Weights
stay excluded; get WavLM from Hugging Face or train the baseline models locally.

Split seed 42 assigns **16 actors/960 files to training**, **four/240 to
validation**, and **four/240 to testing**. Model seed is independent of split
seed. Labels are present in all partitions; actors never cross partitions.
The split is not gender-stratified; both spoken statements occur throughout.
Manifests save absolute paths and need updating if the dataset moves.

The original CNN comparison used four candidates; the later comparison included
WavLM as a fifth. The CNN result remains in `runs/final/`; WavLM's selected
result is in `runs/wavlm_comparison/`. The MFCC option has a trained validation
checkpoint; this project does not report a selected MFCC held-out test result.
Test outcomes have already been inspected; subsequent tuning needs a fresh
evaluation protocol to support an untouched-test claim.

| Artifact | Purpose |
|---|---|
| `data/dataset_path.txt` | Selected dataset folder |
| `runs/analysis/audio_inventory.csv`, `audit.json` | Original-file metadata and health checks |
| Run `splits.json`, `config.json` | File/actor assignments and experiment settings |
| Run `history.json` | Epoch loss, accuracy, validation macro F1, LR and seconds |
| Run `best.pt` | Best validation network weights, buffers and inference metadata |
| WavLM `wavlm_encoder/` | Encoder weights/config and audio processor settings |
| `validation_metrics.json` | Restored best checkpoint's validation evidence |
| Final `selection.json` | Candidates, criterion and selected run |
| Final `test_metrics.json`, `test_predictions.csv` | Aggregate, per-actor and per-recording test evidence |
| `report.html`, `comparison.html`, `prediction.html` | Offline interactive reports |

Open locally generated HTML directly in a browser: Plotly is embedded once in
each original report. The published copies share the same library in _assets/.
Audio, caches, checkpoints and raw run directories remain ignored by Git; the
sanitized runs/published/ snapshot is tracked. Downloading the GitHub repository
does not distribute trained model files or the dataset.

## Verification and illustrative plots

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe Audio_prep.py --audio sample_audio.wav
.\.venv\Scripts\python.exe Fourier_Transformation.py
```

The suite contains **13 tests**: six shared pipeline tests, three WavLM tests
and four app tests. They cover actor splits, filename filtering/duplicates,
features, normalization/checkpoint restoration, a CUDA AMP optimizer step,
frozen/local WavLM, cache invalidation, decoding WAV/WebM, rolling buffers and
input rejection. CUDA checks skip without CUDA. Synthetic tones and a tiny random
encoder check implementation, not emotion accuracy. App tests call callbacks
directly; actual browser permissions, microphones and public access need a demo
check on the devices used.

## Dataset attribution

Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions
in North American English.* PLoS ONE 13(5): e0196391.
[Paper](https://doi.org/10.1371/journal.pone.0196391) ·
[Official dataset and license](https://zenodo.org/records/1188976) ·
[Kaggle speech dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

Dataset license: CC BY-NC-SA 4.0. Scores classify acted speech labels and are not
calibrated confidence or a measurement of a person's internal emotional state.
