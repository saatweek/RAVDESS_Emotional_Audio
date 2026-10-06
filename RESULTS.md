# Original CNN and MFCC experiment results

This records the original four-candidate experiment. Its CNN checkpoint remains
the web app's CNN option. The later frozen WavLM experiment is the current best
reported result: see [WAVLM_RESULTS.md](WAVLM_RESULTS.md). The web demo and study
order are documented in [README.md](README.md) and [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

## Outcome

The selected frequency-preserving CNN achieved **55.42% accuracy (133/240)**,
**0.5324 macro F1**, **0.5508 macro recall**, and **0.5334 weighted F1** on the
held-out test actors. Always predicting any one of the seven larger classes
would score 13.33% accuracy on this test set. Uniform random guessing has
12.5% expected accuracy.

These are actual GPU-trained results, not a promised target or an estimate.
The model is a working educational baseline with substantial room for improvement.

## Dataset audit

The KaggleHub download used dataset version 1. Its archive contains two copies
of the same folder tree; only `audio_speech_actors_01-24` was used.

- 1,440 recording filenames, 24 actors and 88.82 minutes of audio.
- All files decode and contain finite, non-silent samples.
- Original sample rate: 48 kHz; both mono and stereo files occur.
- Original duration: 2.94–5.27 seconds.
- Neutral: 96 recordings; each other emotion: 192 recordings.
- One byte-identical pair: actor 07, happy, normal intensity, statement 02,
  repetitions 01 and 02. Both stay in training. The original named trials were
  retained, so the model sees this training recording twice.
- Zero byte-identical audio pairs cross training, validation or testing.
- One recording contains samples at or above 0.999 amplitude; it was retained.

See `runs/analysis/audio_inventory.csv` for the complete inventory and SHA-256
hashes, and `runs/analysis/audit.json` for the machine-readable audit.

## Evaluation design

The actor split was fixed with Python random seed 42 before model development.

| Partition | Actors | Recordings |
|---|---|---:|
| Training | 01, 02, 03, 04, 05, 07, 08, 09, 10, 13, 15, 17, 20, 21, 22, 23 | 960 |
| Validation | 11, 12, 14, 18 | 240 |
| Test | 06, 16, 19, 24 | 240 |

Every partition contains all eight classes. Speakers never cross partitions.
The split is not gender-stratified. Both lexical statements appear across
partitions, so these results measure generalization to unseen actors reading
familiar statements. Descriptive file-health and metadata checks covered the full
dataset; model comparison used validation predictions only.

## Model comparison

All four candidates in this original comparison used the same actor split. Checkpoints were selected by
validation macro F1, and only the winning candidate was evaluated on test actors.

| Candidate | Model seed | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| CNN with global frequency pooling | 42 | 47.08% | 0.4359 |
| CNN retaining four frequency regions | 42 | **52.50%** | **0.4834** |
| MFCC-statistics MLP | 42 | 44.17% | 0.4452 |
| CNN retaining four frequency regions | 43 | 50.83% | 0.4617 |

The selected model is `runs/cnn_bands_seed42/best.pt`, also copied to
`runs/final/best.pt`. Its best checkpoint is epoch 73; training stopped at epoch
103 after 30 epochs without a better validation macro F1. The two seeds show
some variation; two runs do not establish a reliable distribution of outcomes.

The MFCC model eventually reached about 98% augmented-free training accuracy
while validation accuracy remained much lower. This is evidence of overfitting
to the training data. Its best validation checkpoint, rather than its last epoch,
was used in the comparison. Preserving frequency regions improved the CNN's
validation result relative to global pooling in these experiments.

## Per-emotion test performance

| Emotion | Precision | Recall | F1 | Recordings |
|---|---:|---:|---:|---:|
| Neutral | 0.533 | 0.500 | 0.516 | 16 |
| Calm | 0.735 | 0.781 | 0.758 | 32 |
| Happy | 0.440 | 0.344 | 0.386 | 32 |
| Sad | 0.235 | 0.125 | 0.163 | 32 |
| Angry | 0.611 | 0.688 | 0.647 | 32 |
| Fearful | 0.636 | 0.438 | 0.519 | 32 |
| Disgust | 0.482 | 0.844 | 0.614 | 32 |
| Surprised | 0.629 | 0.688 | 0.657 | 32 |

Calm has the strongest F1. Sad is the weakest class: only 4 of 32 recordings are
correct, with 10 predicted as disgust and eight as calm. Disgust has high recall
but modest precision because other emotions are often assigned to it. These
findings describe the final test results; they were not used for additional tuning.

## Variation across unseen actors

| Actor | Accuracy | Macro F1 |
|---|---:|---:|
| 06 | 51.67% | 0.4700 |
| 16 | 50.00% | 0.4322 |
| 19 | 53.33% | 0.5160 |
| 24 | 66.67% | 0.6341 |

Each actor contributes 60 recordings. Accuracy ranges from 50.0% to 66.7%,
showing why a single overall score is incomplete. Only four actors are held
out, so this is not a precise estimate for arbitrary speakers.

## Artifacts and verification

All plots use Plotly and are saved as standalone, offline HTML files. The reports
include class/split balance, actor coverage, duration distributions, amplitude,
training-only log-mel examples, waveform/FFT/STFT/MFCC views, learning curves,
count and normalized confusion matrices, per-class scores, softmax-score
histograms, candidate comparison, per-actor performance and example prediction.

Finalization saved the selection decision before test inference. Every test
prediction and its eight probabilities are in `runs/final/test_predictions.csv`.
Settings, actor manifests, training histories and best checkpoints are retained.

The environment was verified with PyTorch 2.6.0+cu124 on the NVIDIA GeForce RTX
3050 Ti Laptop GPU (4 GB). At this experiment's completion, six automated tests passed, including an actual CUDA
mixed precision optimizer step. Dependency consistency passed. The dataset audit,
four complete training runs, checkpoint selection, test evaluation and sample WAV
prediction completed successfully. Plotly chart rendering was checked in a browser.
The current suite has 13 tests after adding WavLM and Gradio coverage; this
historical result does not depend on rerunning training for documentation edits.

## Limitations and future experiments

RAVDESS is a small, acted, English-language dataset containing only two statements.
Real conversational recordings, noise, accents and microphones can differ greatly.
The CNN sees a centered four-second segment after trimming. The later Gradio
app repeatedly calls it on rolling windows; the CNN itself has no persistent
conversational memory or long-recording aggregation. Its softmax scores are not calibrated probabilities
of a person's internal feelings.

A future study could use nested actor-group cross-validation, stronger training
augmentation, and controlled fine-tuning of the pretrained speech encoder. Frozen
WavLM has since been implemented and reported in [WAVLM_RESULTS.md](WAVLM_RESULTS.md). Such
experiments should use a fresh evaluation protocol: the reported test outcomes
have now been inspected. High random-file-split scores from other projects are
not directly comparable to these actor-exclusive results.
