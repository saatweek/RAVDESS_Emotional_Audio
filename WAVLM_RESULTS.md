# WavLM emotion classification results

Trained and evaluated on October 6, 2026 using `microsoft/wavlm-base-plus`
through Hugging Face Transformers 4.57.6 and PyTorch 2.6.0+cu124 on the
NVIDIA RTX 3050 Ti Laptop GPU (4 GB).

## Comparison on the existing actor split

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Previously selected CNN | 0.4834 | 55.42% | 0.5324 |
| Frozen WavLM + emotion classifier | 0.7042 | **71.67%** | **0.7038** |

WavLM correctly classified 172 of 240 test recordings, versus the CNN's
133 of 240: an improvement of 16.25 percentage points in accuracy.
`finalize.py` checked that all five candidates used identical manifests,
selected WavLM using validation macro F1, then evaluated that fixed model.
The original CNN results and checkpoint in `runs/final/` remain available.

Training actors: 1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 15, 17, 20, 21, 22, 23.
Validation actors: 11, 12, 14, 18. Test actors: 6, 16, 19, 24.
Training/validation/test sizes: 960/240/240 recordings.

## Configuration and selection

- Frozen pretrained encoder at Hub commit
  `4c66d4806a428f2e922ccfa1a962776e232d487b`; no encoder fine-tuning.
- 16 kHz mono raw waveform, edge-silence trimming, at most four centered seconds.
  Short clips stay unpadded; empty, invalid, silent and very short inputs fail.
- Mean and population standard deviation of the final hidden states:
  1,536 features. Normalization is fitted on training actors only.
- Classifier: 1,536 to 256 linear projection, LayerNorm, ReLU, 35% dropout,
  then eight output logits. Class-weighted cross entropy and AdamW.
- Seed 42 for weights and actor split; classifier batch size 32; initial learning
  rate 0.001; weight decay 0.01; validation plateau learning-rate reduction.
- Best checkpoint: epoch 4, validation accuracy 71.25%, macro F1 0.7042.
  Training stopped at epoch 24 after 20 epochs without a new best macro F1.

## Held-out emotion scores

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Neutral | 0.4063 | 0.8125 | 0.5417 |
| Calm | 0.5833 | 0.6563 | 0.6176 |
| Happy | 0.6000 | 0.4688 | 0.5263 |
| Sad | 0.7895 | 0.4688 | 0.5882 |
| Angry | 0.8378 | 0.9688 | 0.8986 |
| Fearful | 0.8235 | 0.8750 | 0.8485 |
| Disgust | 0.7838 | 0.9063 | 0.8406 |
| Surprised | 1.0000 | 0.6250 | 0.7692 |

Test accuracy by actor: 6 = 70.00%, 16 = 71.67%, 19 = 80.00%, 24 = 65.00%.
This single actor split measures acted RAVDESS labels. It does not establish
accuracy on spontaneous speech or on other actor splits. The frozen WavLM pipeline
improves this experiment, but the comparison also changes features and classifier
architecture; it is not a controlled estimate of pretraining alone. The classifier still overfits
training voices and the scores are not calibrated confidence estimates.

## Saved artifacts and verification

- `runs/wavlm_seed42/`: classifier checkpoint, offline encoder, configuration,
  split manifest, training history and validation metrics/report.
- `runs/wavlm_comparison/`: selected model and encoder, selection record, test
  metrics, per-recording predictions, test report and model comparison report.
- `runs/wavlm_prediction/` and `runs/wavlm_prediction_cpu/`: example WAV
  prediction JSON and charts. Both CPU and CUDA predict `sad` for the unlabeled
  `sample_audio.wav`; this example is a functionality check, not an accuracy claim.

At this experiment's completion, all nine then-existing automated tests passed, including a local WavLM encoder forward pass,
frozen weights, cache revision invalidation, waveform preprocessing, classifier
training, offline checkpoint restoration and existing CNN/MFCC/CUDA checks.
Dependency compatibility checks pass. A previously trained CNN still loads and
predicts successfully through the updated shared command-line workflow.

The current suite contains 13 tests, including four Gradio audio/callback checks.
The app loads this saved encoder and head once and repeats inference on four-second
microphone windows; it does not fine-tune models while visitors speak. Those tests
do not replace browser/device checks or an evaluation on conversational speech.

See [README.md](README.md#train-and-predict-with-wavlm) for training, prediction,
comparison and full reproduction commands. Keep `wavlm_encoder/` beside
`best.pt` when copying a trained WavLM model to another location.
For source explanations and interview study order, see
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) and [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md).
