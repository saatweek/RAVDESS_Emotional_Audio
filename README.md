# RAVDESS Speech Emotion Classification — PyTorch + CUDA

An end-to-end project for eight speech emotions: neutral, calm, happy, sad,
angry, fearful, disgust and surprised. Includes KaggleHub download, dataset audit,
two PyTorch model families, GPU training, actor-exclusive evaluation, WAV prediction,
and offline interactive Plotly reports.

**New to audio or preparing for an interview?** Start with
[INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md). The Python files include step-by-step
comments covering the concepts, array shapes, reasons for choices, alternatives,
and the distinction between tested results and untested possibilities.

**Completed result:** 55.42% test accuracy and 0.5324 macro F1 on 240 recordings
from four unseen actors. The selected CNN was chosen using validation macro F1.
See [RESULTS.md](RESULTS.md) for the experiment comparison and limitations.

## Open the completed results

- Dataset audit and exploration: `runs/analysis/dataset_report.html`
- Final learning curves, confusion matrices and per-emotion scores: `runs/final/report.html`
- Model comparison and per-actor performance: `runs/final/comparison.html`
- Example WAV prediction: `runs/prediction/prediction.html`
- Waveform, Fourier spectrum, spectrogram and MFCCs: `runs/analysis/audio_features.html`
- Two-tone Fourier demonstration: `runs/analysis/fourier.html`
- Selected trained model: `runs/final/best.pt`

Open the HTML files directly in a browser. Plotly is embedded, so the reports work
offline; hover, zoom, toggle series and export individual charts as PNG images.
Large audio files, caches, trained weights and generated reports are kept locally
and excluded from Git. Source code and the written results are versionable.

## Predict an audio file now

Run these commands in the project folder using PowerShell:

```powershell
.\.venv\Scripts\python.exe ravdess.py predict --checkpoint runs/final/best.pt --audio sample_audio.wav --device cuda --output runs/my_prediction
```

Replace `sample_audio.wav` with your WAV path. Use `--device cpu` for CPU inference.
The prediction command prints all eight scores and optionally saves JSON and a
Plotly chart. Silence and empty audio are rejected. Softmax scores are not
calibrated confidence estimates. The selected CNN trims silence and uses a
centered four-second crop or zero padding; it does not analyze an entire long clip.

## Install on another machine

Python 3.12 was used. Create an isolated environment and install the CUDA build
before the other dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The verified machine has an NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM,
and driver 571.96. The prebuilt CUDA 12.4 wheel includes its CUDA runtime.
For a different platform or driver, use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
`requirements-lock.txt` records the exact installed environment; install the
CUDA wheel first, then that file if exact dependency versions are needed.

## Download and inspect the data

```powershell
.\.venv\Scripts\python.exe download_data.py
$datasetPath = Get-Content data/dataset_path.txt
.\.venv\Scripts\python.exe analyze.py --data $datasetPath
```

The downloader calls
`kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")` and
keeps its cache inside `data/`. Kaggle version 1 contains two copies of the data;
the script selects the nested `audio_speech_actors_01-24` copy. Use the saved path,
not the whole cache root, to avoid duplicate filenames. The dataset is public;
if Kaggle requests authentication, use its normal account setup.

The audit validates decoding, finite samples, class/actor counts, sample rates,
channels, duration, amplitude, and byte-identical files. One repeated recording
occurs within actor 07 and stays in training; no byte-identical audio crosses
splits. The inventory and SHA-256 hashes are saved in `audio_inventory.csv`.

## Train, select and evaluate

To reproduce the four experiments and their reports in a new directory:

```powershell
.\.venv\Scripts\python.exe run_project.py --data $datasetPath --output runs/reproduction --device cuda
```

Omit `--data` to download through KaggleHub first. The runner fails immediately
if a step fails. Existing experiment directories are not overwritten.

For a single CNN experiment:

```powershell
.\.venv\Scripts\python.exe ravdess.py train --data $datasetPath --device cuda --model cnn --pool-bands 4 --epochs 150 --patience 30 --skip-test --output runs/new_cnn
```

`--skip-test` reserves test actors during development. The default batch size is
16; reduce it to 8 if GPU memory is insufficient. `--device cuda` fails explicitly
if CUDA is unavailable. `--model mfcc` trains the alternative MFCC-statistics MLP.

Select among completed runs using validation scores, then evaluate the winner:

```powershell
.\.venv\Scripts\python.exe finalize.py --runs runs/new_cnn --output runs/new_final --device cuda
```

To reproduce an existing checkpoint's test evaluation without retraining:

```powershell
.\.venv\Scripts\python.exe ravdess.py evaluate --checkpoint runs/final/best.pt --manifest runs/final/splits.json --device cuda --output runs/reevaluation
```

Manifests contain absolute audio paths. Update them if moving the dataset.
Do not use test scores to tune subsequent experiments and still call the same
test set an untouched evaluation.

## Method

The fixed split seed is 42: 16 actors / 960 files for training, four actors / 240
files for validation, and four actors / 240 files for testing. Model seeds can
change independently using `--seed`; `--split-seed` controls actor assignment.
The two spoken statements occur across all splits; this is an unseen-actor test,
not an unseen-language or unseen-text test. The split is not gender-stratified.

The CNN uses mono 16 kHz audio, silence trimming, four-second crop/padding,
64 log-mel bands, a 512-sample FFT, a 160-sample hop, and per-recording
normalization. Four convolution blocks have 16/32/64/128 channels, batch
normalization, ReLU, max pooling and dropout. The selected architecture retains
four frequency regions before the classifier. The global-pooling baseline retains
one. Time and frequency masks are applied to training batches only.

The alternative MLP uses full trimmed recordings and the temporal means and
standard deviations of 40 MFCCs and their first derivatives (160 features).
Normalization is fitted only to training features and stored as model buffers.
The MLP has 256- and 128-unit hidden layers with layer normalization and dropout.

Both models use class-weighted cross entropy, AdamW, validation-driven learning
rate reduction, gradient clipping, CUDA mixed precision and early stopping.
The best epoch is selected by validation macro F1. Seeds improve repeatability,
but GPU execution is not guaranteed bit-for-bit identical across environments.

Each run saves settings, splits, learning history, validation metrics, and the
best checkpoint. Finalization records the model selection before accessing test
predictions and exports per-recording probabilities and per-actor metrics.
Feature caches are local and keyed by path, file metadata and preprocessing
settings; remove/rebuild the cache after changing the feature extraction code.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe -m pip check
```

Six tests cover repeatable disjoint actor splits, duplicate filenames, stereo
resampling/crop shapes, checkpoint restoration, saved MFCC normalization, silent
input rejection, and an actual mixed precision GPU optimizer step. CUDA checks
skip on machines without CUDA. Full data analysis and training were also run.
The original visualization examples were updated with correct time/frequency
axes, current librosa calls, and offline Plotly exports:

```powershell
.\.venv\Scripts\python.exe Audio_prep.py --audio sample_audio.wav
.\.venv\Scripts\python.exe Fourier_Transformation.py
```

## Dataset attribution

Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions
in North American English.* PLoS ONE 13(5): e0196391.
[Paper](https://doi.org/10.1371/journal.pone.0196391) ·
[Official dataset and license](https://zenodo.org/records/1188976) ·
[Kaggle speech dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

Dataset license: CC BY-NC-SA 4.0. This project uses acted English speech and is
an educational classification baseline, not a validated assessment of a person's
internal emotional state.
