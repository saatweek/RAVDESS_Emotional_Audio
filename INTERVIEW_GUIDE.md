# Understand and explain the project

Use [README.md](README.md) for commands and the reading map, and
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) alongside the source for the purpose of
each statement. This guide explains the ideas and the decisions you need to
defend in an interview. Learn to trace one recording through the system before
trying to memorize implementation details.

## 1. The project in a minute

A useful explanation in your own words:

> This is an eight-class speech emotion project on RAVDESS. I compare a CNN
> on log-mel spectrograms, an MLP on MFCC statistics, and a pretrained WavLM speech
> encoder whose weights I freeze while training an emotion head. I split by
> actor to keep voices separate between training, validation and testing.
> Validation macro F1 selects the checkpoint and candidate. Frozen WavLM achieved
> 71.67% accuracy and 0.7038 macro F1 on 240 recordings from four held-out actors.
> A Gradio app reuses the saved preprocessing and weights for file uploads and
> rolling four-second microphone windows. These results concern acted speech;
> the scores do not measure a person's internal mood.

The original selected CNN scored 55.42% accuracy and 0.5324 macro F1 on those same
test actors. CNN results live in `runs/final/`; the later selected WavLM result
lives in `runs/wavlm_comparison/`. The app offers the trained MFCC baseline too,
but we have not reported a selected MFCC held-out test result.

Be able to explain these five facts before going deeper:

- The target is an **acted recording label**, learned from acoustic samples.
- The three representations have different information and tensor shapes.
- Actor IDs control splits; filename emotion IDs supply targets, never inputs.
- A frozen encoder is pretrained; its emotion head is learned on our training set.
- Live mode repeats classification on windows; latency and generalization remain
  practical limitations.

## 2. Read in stages

The full file order and statement notes are in the walkthrough. Use these passes:

| Pass | Read | Explain without the notes |
|---|---|---|
| First: purpose | README, this guide sections 1–4, result tables | Problem, dataset, inputs, labels, actor split, current result |
| Second: signal path | Fourier demo, Audio_prep, `feature`, `waveform` | Samples → transform/encoder → representation and dimensions |
| Third: learning | Networks, loader, score, train | Trainable state, loss, gradients, validation and best checkpoint |
| Fourth: operation | Finalize, reporting, runner, app | Test reporting and browser audio → saved inference |
| Fifth: evidence | Three test files, both results documents | What tests/results support and what they cannot establish |

For a single line, ask: What object is it operating on? What is its type/shape?
What changes? Does it learn from training data, make a validation decision, or
just transform one input? What later code depends on it? This is more useful
than reciting function names.

## 3. Dataset, labels and leakage

RAVDESS speech has 1,440 named WAV recordings from 24 actors reading two
statements. Eight emotions are represented. Neutral has 96 files; the other
seven classes have 192 each because they include two intensity levels.

Example filename:

```text
03-01-06-01-02-01-12.wav
03 = audio only
01 = speech
06 = fearful (class index 5 after subtracting one)
01 = normal intensity
02 = statement 2
01 = repetition 1
12 = actor 12
```

`records()` validates filenames and creates `{path, label, actor}` metadata.
It does not decode samples. The model receives neither paths nor actor IDs.
The target tells the loss which class is correct; actor tells the split which
recordings must stay together.

The download used Kaggle dataset version 1, with two folder copies. The downloader
selects one complete copy. `records()` rejects duplicate filenames. `analyze.py`
also fingerprints file **bytes**; this finds exact copies with different names,
but not re-encoded or perceptually similar duplicates. One actor-07 duplicate
pair is retained within training. No byte-identical pair crosses the splits.

Split seed 42 shuffles **actors**, not individual files:

| Partition | Actors | Files | Role |
|---|---|---:|---|
| Train | 01, 02, 03, 04, 05, 07, 08, 09, 10, 13, 15, 17, 20, 21, 22, 23 | 960 | Fit network weights and feature normalization |
| Validation | 11, 12, 14, 18 | 240 | Choose epochs/candidates and adjust learning rate |
| Test | 06, 16, 19, 24 | 240 | Score the selected fixed model |

Model seed `--seed` and split seed `--split-seed` serve different purposes.
Changing model initialization should not silently change the test population.
The split is not gender-stratified. Both statements occur across splits, so it
tests unseen voices speaking familiar text, not unseen language or lexical content.

A random file split could put the same actor on both sides. The network could
exploit speaker/recording characteristics, yielding an inflated claim about new
speakers. Actor separation reduces this leakage; it does not guarantee real-world
generalization.

Training feature means/stds for MFCC and WavLM are fitted on training files only.
Applying per-clip CNN normalization to a new clip is different: it uses that
clip's own samples and does not fit across validation/test examples.

Metadata and file-health audit statistics cover all files. Training-only example
plots avoid choosing feature settings by looking at test predictions. Test
outcomes have since been observed for CNN and WavLM; future tuning needs a fresh
evaluation protocol rather than claiming this same test set remains untouched.

## 4. Audio fundamentals

| Term | Meaning | Example here |
|---|---|---|
| Sample | One amplitude measurement | A float in the waveform array |
| Sample rate | Measurements per second | 16,000 Hz after resampling |
| Frequency | Oscillations per second | A 220 Hz tone |
| Waveform | Amplitude versus time | `samples[index]` plotted at `index / sr` |
| FFT | Frequency decomposition of one signal/window | Whole-recording spectrum |
| STFT | FFT repeated in overlapping short windows | Frequency content over time |
| Spectrogram | Matrix of frequency content versus time | Linear-frequency or mel-frequency view |
| Mel filterbank | Overlapping frequency summaries with finer low-frequency resolution | 64 bands for CNN |
| MFCC | Cosine-transform coefficients of log-mel spectral shape | 40 coefficients before statistics |
| Tensor | Multidimensional numeric array | `[batch, channel, frequency, time]` |

A four-second 16 kHz recording contains 64,000 samples. Sample rate is not pitch:
a 220 Hz wave sampled 16,000 times per second is still a 220 Hz wave. Resampling
computes a new representation at another rate; changing a file header alone
changes playback speed and pitch. At 16 kHz, the highest representable frequency
is about 8 kHz.

`Fourier_Transformation.py` adds 30 Hz and 70 Hz sine waves and recovers peaks
with an FFT. `Audio_prep.py` illustrates real audio. Neither trains a model or
produces the exact CNN input; the latter uses a full recording without edge
trimming/cropping and plots 20 MFCCs using librosa defaults.

The STFT settings are a 512-sample window (32 ms at 16 kHz) and 160-sample hop
(10 ms). Larger windows improve frequency resolution while reducing time
localization; smaller hops add more overlapping columns and computation.
These settings are practical choices, not proven optima.

Power is squared magnitude. `power_to_db` uses a factor of 10 on a log ratio;
`amplitude_to_db` uses 20. With `ref=np.max`, 0 dB means the strongest cell
in that particular representation, not a universal absolute sound level.
Do not call FFT magnitude a power spectrum or MFCC coefficient index a frequency.

## 5. The CNN recording path

Follow `ravdess.feature(path, config)` with a saved CNN config:

1. Load/resample to mono 16 kHz and reject invalid or silent audio.
2. Trim quiet leading/trailing frames with a relative 35 dB threshold. This
   removes edges, not interior pauses, noise or all nonspeech.
3. Center-crop longer trimmed clips to 64,000 samples. Right-pad shorter ones
   with zeros. Fixed length makes stacking and minibatches simple.
4. Compute 64 mel-band power values per STFT frame.
5. Convert to peak-relative log power.
6. Subtract that matrix's mean and divide by its std plus a small epsilon.
7. Cast to float32 and add one channel axis: `[1, 64, 401]`.

With centered librosa framing, 64,000 samples and hop 160 produce 401 columns.
These numbers are fed directly into the network. The CNN never reads a Plotly
heatmap or its colors. Per-clip normalization reduces recording-scale effects
but removes absolute amplitude cues, a tradeoff for acted intensity.

Four convolution blocks each contain convolution, batch normalization, ReLU,
max pooling and dropout. The shape trace is:

```text
[B, 1,   64, 401]
[B, 16,  32, 200]
[B, 32,  16, 100]
[B, 64,   8,  50]
[B, 128,  4,  25]
[B, 128,  4,   1]  adaptive average pooling, selected variant
[B, 512]           flatten
[B, 8]             dropout + linear classifier
```

A learned 3×3 convolution searches for local time/frequency patterns. Padding
preserves the spatial dimensions before pooling. Max pooling reduces both axes.
Adaptive average pooling averages time while keeping four frequency regions.
The original global-pooling baseline retains one region and flattens to 128.
The four-region variant performed better on validation in these runs; that
does not prove every audio CNN must retain those exact regions.

BatchNorm uses batch statistics during training and stored estimates at
evaluation. ReLU introduces a nonlinearity; stacked linear mappings alone would
remain one linear mapping. Dropout removes units/maps randomly during training
and is disabled at evaluation. These mechanisms do not guarantee no overfitting.

CNN training additionally zeros six mel bands and 20 time frames (~0.2 seconds),
with one pair of mask positions shared within each batch. Zero means the
normalized feature mean, not literal acoustic silence. Validation/inference
have no masks. Random crops, additive noise and pitch/time-stretch augmentation
were not part of the completed experiments.

## 6. The MFCC + neural network path

The MFCC branch loads and trims the full recording. It does **not** apply the CNN
four-second crop/padding. It computes 40 MFCC coefficients over 64 mel bands and
their first-order delta coefficients.

For each of the 40 coefficients, it stores:

- Mean over time.
- Population standard deviation over time.
- Mean of the delta over time.
- Population standard deviation of the delta over time.

Concatenation gives 160 features. This compresses variable-length audio into a
fixed vector cheaply, but discards the order in which events occurred.
Coefficient index is a cepstral dimension, not Hertz or pitch.

`EmotionMLP` first uses saved feature-wise means/stds fitted on training actors.
These are **buffers**, which move with the model and appear in its state dictionary
but are not learned parameters. The network is:

```text
[B,160] → Linear(160,256) → LayerNorm → ReLU → Dropout(0.35)
        → Linear(256,128) → LayerNorm → ReLU → Dropout(0.35)
        → Linear(128,8)
```

LayerNorm normalizes activations within an example. It differs from both the
input's train-fitted normalization and CNN BatchNorm. The MFCC model overfit:
training accuracy became much higher than validation accuracy. We use the best
validation epoch, not the last one. It serves as a feature-based baseline and
a web app option, not a reported winning test model.

## 7. The WavLM path and Hugging Face

WavLM is a pretrained speech encoder. Hugging Face Transformers supplies the
architecture, loading APIs and audio processor. Our code supplies waveform
validation, revision handling, freeze/pooling choices, normalization, an emotion
head and the training/evaluation pipeline.

The completed run uses `microsoft/wavlm-base-plus` at commit
`4c66d4806a428f2e922ccfa1a962776e232d487b`. A Hub `main` reference can move,
so the training initializer resolves a commit and loads config, processor and
weights from that revision. It exports an offline encoder alongside `best.pt`.

The preprocessing in `wavlm.waveform()`:

1. Read original-rate audio with channels preserved for validation.
2. Reject invalid/silent samples, then downmix to mono and resample to 16 kHz.
3. Trim edges and take at most four centered seconds.
4. Keep short clips unpadded. Reject fewer than 400 samples (25 ms) for base-plus.

The saved audio feature extractor prepares tensors and attention-mask/normalization
settings. It is not a text tokenizer. This base-plus processor's saved settings
have `do_normalize=False`; our code does not add an assumed processor
normalization step. The emotion head has its own train-fitted normalization.

The encoder maps samples to contextual hidden states:

```text
[N samples] → processor → input_values [1,N]
             → frozen WavLM → [1,T,768]
             → select recording → [T,768]
             → time mean [768] + population time std [768]
             → concatenate → [1536]
```

The encoder downsamples time, so `T` is not `N` or the CNN's 401 columns.
Mean pools the contextual representation; std adds variation across time.
Pooling loses temporal order despite using contextual states. One unpadded clip
per call avoids pooling padding and keeps extraction memory small.

Three different mechanisms keep inference appropriate:

- `.to(device)`: place tensors/weights on CPU or CUDA.
- `.eval()`: disable training-time stochastic behavior.
- `.requires_grad_(False)` and `inference_mode()`: freeze encoder parameters
  and avoid constructing an autograd graph.

`eval()` alone does **not** stop gradient computation. The frozen encoder is
outside the trainable head and outside its optimizer. After extracting/caching
features, training updates only:

```text
[B,1536] → train-fitted normalization
         → Linear(1536,256) → LayerNorm → ReLU → Dropout(0.35)
         → Linear(256,8)
```

Cached features remain valid across epochs because encoder weights do not
change. An unfrozen encoder would need recomputation as its weights change.
New uploaded/microphone audio is encoded at prediction time; the app doesn't
look for it in the training cache.

This is **transfer learning with a frozen encoder**, not encoder fine-tuning.
Pretraining provides speech representations, not our eight-class mapping.
Fine-tuning could help but would require gradients, more memory, careful learning
rates and new evaluation. We have not tested full encoder fine-tuning.

## 8. What actually happens in training?

Read `loader()`, `score()`, then `train()`. The train routine:

1. Validates positive limits; seeds Python, NumPy and PyTorch.
2. Creates a new run directory, parses files and saves actor manifests.
3. For WavLM, loads/freezes/exports the encoder and records resolved settings.
4. Prepares training and validation features. Test feature extraction is excluded
   when developing with `--skip-test`.
5. Builds the trainable network. Fits MFCC/WavLM input buffers on training
   features only; clamps stds away from zero.
6. Counts training labels and weights cross entropy inversely by class frequency.
7. Creates AdamW, a validation-driven scheduler and CUDA gradient scaler.
8. Runs epochs of minibatch updates, evaluates validation, and saves strict
   improvements in validation macro F1.
9. Stops after consecutive nonimproving epochs reach patience; restores the best
   checkpoint and writes validation metrics/report.

The DataLoader stacks features in CPU RAM and transfers only a minibatch.
Training shuffles; evaluation preserves manifest order so filenames still align
with predictions. This dataset is small; a larger one would need a lazy dataset
and possibly parallel loading. Feature cache keys use file metadata/settings,
not automatic hashes of source code. Editing extraction logic requires deliberate
cache invalidation.

For each minibatch:

```text
clear previous gradients → forward logits → weighted loss
→ scaled backward gradients → unscale → clip norm at 5
→ optimizer step → update scaler
```

**Logits** are raw class scores. Cross entropy includes log-softmax, so the
network should not apply softmax before that loss. Display uses softmax and
argmax; softmax preserves the winning class while normalizing the scores.

Neutral has half as many recordings. The formula `N / (8 * class_count)`
gives its mistakes twice the relative weight. Class weighting differs from
oversampling: it changes loss contributions rather than drawing extra examples.

AdamW adapts updates and applies weight decay (0.01). Learning rate starts at
0.001. Gradient clipping limits the global parameter-gradient norm after
unscaling. CUDA autocast chooses lower precision for suitable head/CNN operations;
GradScaler helps prevent small gradients underflowing. CPU uses float32, and the
frozen WavLM extraction remains float32.

The scheduler halves LR after a validation macro-F1 plateau with patience 5.
Early stopping uses a separate, larger experiment patience. The scheduler can
reduce step size without terminating training. Best checkpoint means best
validation macro F1, not lowest training loss or highest test accuracy.

An epoch sees all 960 training examples. Batch 16 yields 60 updates; batch 32
yields 30. `--epochs` is a maximum: WavLM's best was epoch 4 and it stopped
at 24 with patience 20. Selected CNN's best was 73 and it stopped at 103 with
patience 30.

Training accuracy includes dropout, changing weights and CNN masking; validation
uses a fixed model in eval mode. Validation accuracy can exceed training accuracy
without leakage. Training loss is class-weighted and accumulated from minibatch
means, while validation loss is unweighted per-recording cross entropy; their
numerical gap is not a clean same-objective comparison.

`best.pt` saves parameters, buffers, architecture/preprocessing metadata and
selected epoch. It does not save optimizer/scheduler/scaler/RNG state for an exact
training resume. WavLM additionally needs the exported encoder/processor directory.

## 9. Metrics, selection and results

Accuracy is fraction correct. For one class, precision asks how many predicted
examples of that class are correct; recall asks how many true examples were found.
F1 is their harmonic mean. Macro F1 averages the eight class F1s equally, keeping
the smaller neutral class visible. Weighted F1 instead weights by support.
Undefined precision/F1 uses zero in this implementation.

Confusion matrix rows are true classes; columns are predictions. Row-normalized
diagonal entries are recalls. The highest softmax score is not calibrated
confidence: high-scoring mistakes remain possible.

Original validation comparison:

| Candidate | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| CNN, global frequency pooling, seed 42 | 47.08% | 0.4359 |
| CNN, four frequency regions, seed 42 | 52.50% | 0.4834 |
| MFCC MLP, seed 42 | 44.17% | 0.4452 |
| CNN, four frequency regions, seed 43 | 50.83% | 0.4617 |
| Later frozen WavLM + head, seed 42 | **71.25%** | **0.7042** |

`finalize.py` requires identical train/validation/test manifests, picks by
validation macro F1, saves `selection.json`, then copies and tests the winner.
An exact tie keeps the first candidate; test performance never breaks it.

| Reported selected model | Correct test recordings | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Original four-region CNN | 133/240 | 55.42% | 0.5324 |
| Later frozen WavLM + head | 172/240 | 71.67% | 0.7038 |

The improvement is **16.25 percentage points**, not 16.25% relative improvement.
We changed both representation and classifier, so the comparison does not isolate
pretraining alone as a causal explanation. WavLM's early best epoch and subsequent
training improvement do not mean it generalizes perfectly.

For the CNN, sad recall was 4/32 and calm had the strongest F1. For WavLM, angry,
fearful and disgust had stronger F1 than happy, neutral and sad. Exact per-emotion
tables are in the result documents. WavLM's per-actor accuracy is 70%, 71.67%,
80% and 65%; four actors are too few for a population guarantee.

A majority-class guess achieves 13.33% here; uniform random guessing has expected
12.5% accuracy. Better than these baselines is useful evidence, not proof of
reliability. Two CNN seeds and one split do not estimate robust variability.

The unlabeled `sample_audio.wav` example checks operation, not accuracy. A
model's prediction is not its own ground truth. CPU/CUDA may agree in argmax while
differing slightly in scores; do not promise bit-identical results across devices.

## 10. Browser app and hosting

`app.py` uses Gradio for the page, event transport and queue. It loads all three
checkpoints once, validates label order and warms them with a synthetic tone.
Warm-up checks readiness and reduces some first-use overhead; it does not check
emotion recognition accuracy.

Upload flow:

```text
browser file → Gradio temporary filepath → bundled FFmpeg
→ mono 16 kHz float32 samples → input validation
→ temporary float WAV → saved model-specific extractor
→ batch dimension → saved network → eight softmax scores → browser label
```

The intermediate WAV lets all three branches reuse exactly the same path-based
preprocessing contract as the CLI. It is removed at the end of each prediction.
Decode limits are 25 MB, 30 seconds and a 20-second FFmpeg timeout. An extra decoded
second detects overlong uploads before rejecting them.

Microphone flow:

```text
browser records chunks → filepath decoder → append per-session samples
→ retain last 64,000 received samples → wait until full / skip very quiet window
→ same prediction service → display updated scores
```

Gradio uses filepath input for live recording here; the callback turns it into
a `(16000, samples)` pair. A reusable helper also supports integer PCM,
stereo channel averaging and resampling. `gr.State` holds each visitor's
buffer separately. Models/lock are shared, not audio history.

Requests are configured around one-second stream intervals. First output needs
four seconds of received context plus transport/processing time. The app keeps
recording until the user stops. The stream's `time_limit=30` is a Gradio queue
scheduling budget, not a 30-second microphone auto-stop.

A fixed RMS threshold suppresses very quiet windows. There is no learned VAD,
noise classifier, smoothing, transcript context or continuous recurrent memory.
The latest received window may lag wall-clock speech if work is queued; there
is no explicit stale-chunk dropping. Avoid claiming hard real-time guarantees.

The trained WavLM package is also published on the Hugging Face Hub. That stores
downloadable model assets; the laptop still hosts Gradio. download_model.py pins
a Hub commit and checks declared hashes before the normal local loader is used.
The package includes the frozen encoder and custom head, so no retraining is
needed for WavLM inference. It is not a standard Transformers classification
export and does not require executing downloaded Python code.

Uploads and streams have separate scheduler groups. A shared lock serializes
preprocessing and inference to limit GPU overlap; it may also reduce throughput.
The Gradio queue allows up to 16 pending events; stream concurrency is configured
at four. This is a small laptop demonstration, not a scalability benchmark.

The server binds to laptop loopback at the selected port (default 7860).
`--share` supplies a temporary public HTTPS relay link. Friends use that link:
their own 127.0.0.1 points to their own device. Microphone use requires browser
permission and an appropriate browser context; the HTTPS share URL supports it.
The laptop must remain awake/online with the server running.

There is no prediction history or timing log. To add meaningful latency reporting
later, separate decode/preprocessing time, encoder/head inference, queue wait and
browser round trip. A larger observed delay does not necessarily mean a larger
model. CUDA kernel timing also needs synchronization or CUDA events.

Model changes clear scores but keep audio context. Starting/clearing a recording
resets its state; stopping resets the buffer and preserves the last prediction.
Cancellation does not undo a callback already executing. Gradio's cached files
are cleaned periodically, while our per-inference WAV is removed immediately.

## 11. Tests and proof

There are 13 tests: six shared pipeline, three WavLM and four Gradio checks.

| Evidence | Supports | Does not establish |
|---|---|---|
| Actor split/parser tests | Repeatability, disjoint groups, metadata handling | Generalization accuracy |
| Tone/shape and normalization round trips | Dimensions, resampling and saved state | Emotion labels for synthetic tones |
| CUDA AMP step | Actual GPU forward/backward/update path | GPU correctness on every platform |
| Tiny local WavLM and cache tests | Freeze semantics, offline restoration, revision cache invalidation | Quality of downloaded pretrained representations |
| App decoder/callback tests | WAV/WebM decoding, rolling buffers, resets/rejection | Real microphone permissions, browser timing or multiuser network behavior |
| Held-out actor evaluation | Scores on this fixed acted speech population | Spontaneous speech, arbitrary accents/noise or calibrated mood estimates |

The tiny WavLM test has hidden size eight and uses randomly initialized local
weights: 16 pooled features rather than 1,536. This avoids downloading a large
model to test integration. Testing `eval()` still permits gradients in the head;
freezing the encoder is a separate property.

No retraining is needed after comment/documentation changes. Syntax checks,
executable AST comparison and the existing tests can verify the annotations
did not change behavior. Check actual audio/browser behavior on demo devices.

## 12. Practice questions

**Why use three models?** To compare a learned local spectrogram model, a compact
handcrafted summary baseline and a pretrained speech representation under the
same actor split and evaluation criterion.

**Why a spectrogram instead of raw samples for the CNN?** It exposes evolving
frequency structure in a compact 2D array. Raw-waveform learning is possible,
but its architecture and data requirements differ. WavLM supplies that learned
raw-waveform encoder path.

**Why 16 kHz and four seconds?** Compact speech processing and manageable memory.
Four seconds fits much of this dataset after trimming. It can omit useful audio;
the choices were not established as universally best.

**Why not crop MFCC too?** Statistics already map variable-length recordings
to a fixed vector. Keeping the full trimmed clip preserves more summary evidence,
though it makes long-upload comparisons differ from CNN/WavLM.

**Why mean and std pooling for WavLM?** Fixed-size vectors with average context
and temporal variation, without padding. It discards order and was not compared
against all layer/pooling alternatives.

**Why freeze rather than fine-tune?** Fewer trainable parameters, cached features
and smaller memory requirements on the 4 GB GPU. Fine-tuning remains an untested
tradeoff; freezing is not always the accuracy-maximizing choice.

**What prevents leakage?** Whole-actor splits, duplicate checks, train-only
normalization, validation-based selection and deliberate test access. These
controls don't resolve every external-data or domain-shift question.

**What does `model.eval()` do?** Changes dropout/BatchNorm behavior. It does
not freeze parameters or disable autograd; inference mode handles graph tracking.

**Why no softmax in the network?** Cross entropy operates on logits and includes
log-softmax. Use softmax when reporting eight relative scores.

**Why use macro F1?** Equal class importance despite neutral's smaller support;
accuracy alone can hide a weak class.

**Why does epoch 4 win WavLM training?** Later fitting of training data did not
improve validation macro F1. Restoring the best checkpoint preserves the selected
model rather than the later overfitted weights.

**Does the web app train on visitors?** No. It loads inference checkpoints and
repeats forward passes; no optimizer step or persistent visitor learning exists.

**Can it work on any device?** The model runs on the laptop; a visitor needs a
compatible browser/network and permitted audio capture. The share link enables
remote access, while localhost does not. Real device checks are still necessary.

**Why not claim it detects someone's true mood?** Acted labels, a small dataset,
uncalibrated scores and domain shift do not justify that claim.

**What would you improve first?** Establish a fresh actor-group evaluation,
measure latency on intended devices, test noisy/spontaneous recordings, then
compare pooling/augmentation/fine-tuning using validation. Prioritize evidence
before architecture complexity.

**How do you describe coding assistance?** Be accurate about the help you used.
Explain the design, follow the data through the source, identify limitations and
show verification. Understanding and ownership should be demonstrated by what
you can explain and change, not by claiming to have written every line unaided.

## 13. Final self-check before an interview

- Trace one filename to target/actor metadata without passing its label as input.
- Draw all three input/feature/head shapes and explain which weights train.
- Explain class weighting, buffers, train/eval modes and one optimizer update.
- Identify what validation chooses and when test recordings are first scored.
- Show a saved configuration, history, selection record and a real error.
- Start/check the Gradio server; test uploads and your intended microphone/browser.
- Explain the difference between a four-second rolling window and guaranteed
  real-time speech tracking.
- State the current WavLM result and historical CNN result with their denominator.
- Name what the synthetic tests prove and where real data/device evidence is needed.
- Explain a sensible next experiment without calling an inspected test set untouched.

If one step is hard to explain, return to its numbered source section in
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md), then describe its input, operation,
output and reason out loud.
