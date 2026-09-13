# Understand and explain this project

Start with the problem, then follow one recording through the code. You do not
need to master all of signal processing before Monday. Aim to explain what the
system learns, how the input is represented, how evaluation avoids leakage, and
where the result is weak. Use the code comments for the details you cannot yet
explain in your own words.

If you have only 15 minutes for a first pass, learn these five facts:

- It predicts eight acted speech emotions from audio, not a transcript.
- The selected CNN receives a normalized 64-by-401 log-mel matrix per clip.
- Actors are separated: 16 train, four validation, four test.
- Validation chose the winner; its final test result is 133/240 correct and
  0.5324 macro F1. Sad is its weakest class.
- Several settings are practical baseline choices, not proven optima; the result
  does not establish reliability on ordinary conversations.

## 1. What problem are we solving?

**Input:** a recorded spoken sentence. **Output:** one of eight recorded emotion
labels: neutral, calm, happy, sad, angry, fearful, disgust or surprised.

This is **supervised multiclass classification**: training examples have known
labels, and each recording has one target class. It is not speech recognition
(turning speech into words), speaker identification (recognizing who spoke), or
audio generation. The model predicts an acted dataset label, not someone's
verified psychological state.

RAVDESS speech has 1,440 named recordings from 24 actors reading two statements.
The filename encodes the target label. For example:

```text
03-01-06-01-02-01-12.wav
│  │  │  │  │  │  └─ Actor 12
│  │  │  │  │  └──── Repetition 1
│  │  │  │  └─────── Statement 2
│  │  │  └────────── Normal intensity
│  │  └───────────── Emotion 6: fearful
│  └─────────────── Speech
└────────────────── Audio only
```

The network receives features derived from audio samples. It does **not** receive
the filename, actor ID, or emotion ID as an input. The label is used to measure
training error; actor ID determines the data split.

**Check yourself:** Why would feeding the emotion number from the filename to the
network make the experiment meaningless? It would reveal the answer at input
time, and ordinary new audio does not come with that answer.

## 2. Audio basics, without assuming prior knowledge

| Term | Meaning | In this project |
|---|---|---|
| Sample | One measured amplitude value | A floating-point number after loading |
| Sample rate | Number of samples per second | Standardized to 16,000 Hz |
| Amplitude | Signal value at an instant | Related to signal strength, not an emotion label |
| Frequency | Oscillations per second, measured in Hz | Describes components of the sound |
| Pitch | Perceived highness/lowness of a sound | Related to periodicity; not explicitly extracted here |
| Waveform | Amplitude plotted against time | First plot in `Audio_prep.py` |
| Spectrum | Strength of frequency components | FFT plot |
| Spectrogram | Frequency content changing over time | STFT and log-mel representations |
| Tensor | A multidimensional numerical array | PyTorch input, weights and outputs |

A one-second recording at 16 kHz has 16,000 samples. A four-second recording has
64,000. **Sample rate is not pitch**: recording a 220 Hz tone at 16 kHz means taking
16,000 measurements per second of a signal oscillating 220 times per second.

Resampling computes a new sequence representing approximately the same signal
at another sample rate. Merely changing the rate in a file header would change
playback speed/pitch. Resampling to 16 kHz also limits representable frequencies
to below about 8 kHz; retaining 48 kHz could preserve higher-frequency details.

Open `runs/analysis/fourier.html`. Two tones at 30 Hz and 70 Hz combine into one
waveform; its Fourier spectrum reveals both. Real speech contains many changing
components, so a single FFT of the entire recording loses useful timing.

## 3. Follow one WAV through `feature()`

```mermaid
flowchart LR
    A[WAV samples] --> B[Mono and 16 kHz]
    B --> C[Trim quiet edges]
    C --> D[Crop or pad to 4 seconds]
    D --> E[Windowed Fourier power]
    E --> F[64 mel bands]
    F --> G[Log scale and normalize]
    G --> H[CNN]
    H --> I[8 scores]
    I --> J[Predicted emotion]
```

Read `ravdess.py: feature` alongside this section.

**Load and standardize.** `librosa.load(..., sr=16000, mono=True)` converts different
recordings to a common representation. The audit found both mono and stereo
source files. Mono simplifies the task; spatial stereo information is not a
focus of this classifier.

**Trim quiet edges.** `top_db=35` removes leading/trailing frames sufficiently quiet
relative to that clip's maximum RMS. It does not remove background noise or
interior pauses. A quiet spoken onset could be removed, so this is a tradeoff,
not automatically an improvement. See the [librosa trim documentation](https://librosa.org/doc/0.11.0/generated/librosa.effects.trim.html).

**Make lengths consistent.** The CNN branch keeps the center four seconds of a
long trimmed clip, or appends zeros to a shorter one. This makes batching simple.
It may miss an emotional event near an edge of a long recording. A future version
could classify several windows and aggregate their predictions.

**Look at short windows.** Speech changes over time. The STFT analyzes overlapping
windows; here 512 samples span 32 ms, and a 160-sample hop advances 10 ms. Shorter
windows localize changes in time better, while longer windows can separate close
frequencies better. The FFT bin spacing here is 16,000/512 = 31.25 Hz; this is not
the spacing of the final mel bands.

**Build mel features.** Squared Fourier magnitudes describe power. Overlapping mel
filters combine that power into 64 bands. Log conversion compresses its dynamic
range. The output is a numerical time-frequency matrix, not an RGB image. With
librosa's default centered framing, the fixed input produces 401 time columns.
See the [mel-spectrogram API](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html).

**Normalize.** Subtract each recording's feature mean and divide by its standard
deviation. This can reduce sensitivity to recording level, but removes absolute
loudness information. The model still sees relative patterns across time and
frequency. A tiny added number prevents division by zero.

The exact constants—16 kHz, four seconds, 64 bands, 35 dB—were practical baseline
choices. We did not run an experiment proving each one better than all alternatives.

## 4. What does the CNN actually do?

Read `EmotionCNN` in `ravdess.py`. A **convolution** moves a learned small filter
across the feature matrix. Training changes filter values so combinations of local
patterns help predict labels. We do not manually assign a filter to detect anger.

Each block contains convolution, batch normalization, ReLU, max pooling and
dropout. Normalization helps control activation scales; ReLU introduces a
nonlinear transformation; pooling reduces size; dropout randomly removes parts
of the representation during training to discourage dependence on a few features.
None guarantees better generalization on its own.

For the selected model, the shape trace is:

| Stage | Tensor shape | How to read it |
|---|---|---|
| One clip before batching | `[1, 64, 401]` | One input channel, mel bands, time frames |
| Batch | `[16, 1, 64, 401]` | Sixteen recordings |
| Block 1 | `[16, 16, 32, 200]` | Sixteen learned feature maps |
| Block 2 | `[16, 32, 16, 100]` | More channels, smaller spatial dimensions |
| Block 3 | `[16, 64, 8, 50]` | Same pattern |
| Block 4 | `[16, 128, 4, 25]` | Four frequency regions remain |
| Average over time | `[16, 128, 4, 1]` | Keep coarse frequency location |
| Flatten | `[16, 512]` | One feature vector per recording |
| Linear classifier | `[16, 8]` | Eight logits per recording |

Here the first `16` is batch size; the second dimension after a convolution is
learned feature channels, not stereo channels. A **logit** is a raw score that can
be negative or positive. It need not sum to one. Softmax converts logits to scores
summing to one; argmax selects the highest-scoring class.

The global-pooling baseline also averages frequency into one region. Retaining
four frequency regions achieved higher validation macro F1 in our comparison.
The final time average still loses some detailed ordering; this is not a full
sequence model. An LSTM or attention layer could model timing differently, but
neither was implemented or evaluated here.

## 5. What was the MFCC alternative?

MFCC means **mel-frequency cepstral coefficient**. MFCCs apply a cosine transform
to a log-mel representation, producing a compact description of spectral shape.
An MFCC axis is a coefficient index, not a frequency axis.

Our alternative extracts 40 MFCCs and their first temporal derivatives, then
takes each one's temporal mean and standard deviation: `40 × 4 = 160` features.
This branch uses the full trimmed clip, without the CNN's four-second crop.
It learns a multilayer perceptron (MLP) on those summary numbers.

Its normalization is different from the CNN's: each of the 160 features gets a
mean/std fitted over **training examples only**. The stored values are reused at
validation and prediction time. Fitting them on all examples would leak held-out
data into preprocessing. They are PyTorch **buffers**, saved with the model but
not updated by the optimizer.

The MLP is fast and compact, but summaries discard temporal ordering. It reached
very high training accuracy without comparable validation improvement. This is
overfitting, not evidence that more training would necessarily solve the problem.

## 6. How does training learn?

Read `train()` in `ravdess.py`. A **parameter** is learned, such as a convolution
weight. A **hyperparameter** is chosen externally, such as learning rate or batch
size. An **epoch** is one pass through the training set. A **batch** is the smaller
group used for an optimizer update: 960 files / 16 = 60 batches per CNN epoch.

The loop is:

1. Move the batch to the GPU. Model and input tensors must be on the same device.
2. For the CNN, hide a small time strip and frequency strip. This teaches the
   model to cope with missing local information. It is applied only in training.
3. Clear old gradients, because PyTorch normally accumulates them.
4. Run the forward pass to obtain logits.
5. Compute cross entropy: penalize assigning low probability to the true label.
6. Backpropagate using the chain rule to compute how weights affect the loss.
7. Clip excessive gradient magnitude and let AdamW update the weights.
8. After all batches, evaluate on validation actors and save the best checkpoint.

For an unweighted single example, cross entropy is `-log(p_true)`. A true-class
probability of 0.8 produces a smaller loss than 0.1. The actual PyTorch loss
accepts **logits** and handles log-softmax internally; we do not apply softmax
before `CrossEntropyLoss`.

Neutral has half as many training examples as each other class. The loss gives
neutral errors twice the relative weight. Oversampling could also address that
imbalance; we used weighting. These methods change learning priorities, not the
number of test examples or how accuracy is counted.

AdamW adjusts updates using gradient history and applies weight decay. Learning
rate sets update scale. When validation macro F1 stalls, the scheduler lowers
that rate. Early stopping later ends training after a specified number of epochs
without a new best score. Our winner peaked at epoch 73 and stopped at 103.

The saved checkpoint contains weights, preprocessing, architecture settings and
some run metadata. It does not contain everything required to resume training
exactly at the next step: optimizer, scheduler, scaler and RNG states are missing.

## 7. What role did the GPU play?

The NVIDIA RTX 3050 Ti has 4 GB VRAM. PyTorch used CUDA for neural-network
operations; librosa feature extraction ran on the CPU and was cached locally.
Only batches needed to be sent to the GPU. We did not benchmark an exact speedup
over CPU training, so do not invent a multiplier.

Automatic mixed precision chooses lower precision for suitable operations, which
can reduce memory use and improve throughput. Gradient scaling helps small
float16 gradients survive numerically. We unscale before clipping, then update
weights. Full float32 is a simpler alternative and is used for CPU execution
here. See [PyTorch's mixed precision examples](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html).

The GPU changes computational cost; it does not automatically make predictions
more accurate. The completed CPU/GPU sample predictions differed in scores by
less than 0.000001 and chose the same class.

## 8. Why the actor split is central

The split is 16 training actors, four validation actors and four test actors:
960/240/240 recordings. Model seed and actor-split seed are independent.

Training data changes model weights. Validation data selects checkpoints and
candidate models. Test data estimates performance after that selection. Splitting
individual recordings randomly could put the same voice in training and testing,
which would not support a claim about performance on unfamiliar voices.

There are several kinds of leakage to distinguish:

| Risk | What this project does |
|---|---|
| Same actor across partitions | Groups files by actor before splitting |
| Duplicate folder copies | Uses one Kaggle subtree and rejects duplicate filenames |
| Identical bytes under different names | Audit checks hashes; one pair stays in training |
| Preprocessing fitted on test data | MLP scaling fits training only; CNN scaling is per clip |
| Choosing the model with best test score | Chooses validation winner before test inference |

The audit does not prove absence of all near-duplicate or re-encoded sound. The
split is not gender-stratified and both spoken statements occur across partitions.
It is an unseen-actor evaluation, not an unseen-text or real-world benchmark.

Once test errors have been inspected, further tuning around those errors would
use that knowledge. A future study needs a fresh protocol, such as nested actor
group cross-validation. We did not run cross-validation in this project.

## 9. Explain the results numerically

**Accuracy:** 133 correct recordings / 240 = **55.42%**. A constant prediction of
one of the larger classes gives 32/240 = 13.33%; uniform random guessing has
12.5% expected accuracy. Beating these references is useful but not enough for a
reliable product.

**Precision:** among predictions of an emotion, how many are correct?
**Recall:** among true recordings of that emotion, how many were found?
**F1:** `2 × precision × recall / (precision + recall)`.
**Macro F1:** compute F1 separately for all eight classes and take their plain
average. It is not the F1 calculated from average precision and average recall.
See the [F1 definition and averaging options](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Use sad as a concrete example: the model correctly found 4 of 32 sad clips and
predicted sad for 17 clips total. Precision is 4/17 ≈ 0.235, recall is 4/32 = 0.125,
and F1 is about 0.163. This weak class matters even though overall accuracy is
55.4%. Final macro F1 is **0.5324**. Calm has the highest class F1, about 0.758.

Open `runs/final/report.html`. Confusion matrix rows are true labels and columns
are predictions. Ten sad recordings were called disgust and eight were called
calm. A row-normalized diagonal is per-class recall. Do not read those cells as
precision.

Model comparison used validation, not test, scores:

| Candidate | Validation macro F1 |
|---|---:|
| Globally pooled CNN, seed 42 | 0.4359 |
| Four-region CNN, seed 42 | **0.4834** |
| MFCC MLP, seed 42 | 0.4452 |
| Four-region CNN, seed 43 | 0.4617 |

Only the validation winner was tested. Results vary by test actor from 50.0% to
66.7% accuracy. Four people are too few to make a strong population-wide claim.

## 10. Choices, alternatives and what the evidence supports

| Choice | Why it was used | Alternative and its tradeoff | Actually compared? |
|---|---|---|---|
| Log-mel CNN | Learns local time/frequency patterns with a small model | Raw-waveform model learns its own front end but adds modeling demands | Raw waveform: no |
| Four frequency regions | Retains coarse spectral location | Global pooling is smaller but loses location | Yes; four regions won validation here |
| MFCC MLP | Fast conventional feature baseline | Sequence features retain order but add complexity | Yes; lower validation F1 |
| 16 kHz mono | Compact, consistent inputs | 48 kHz/stereo retains more detail at added cost | No ablation |
| Four-second center crop | Simple fixed batching | Random/multiple crops preserve different segments, at added logic/cost | No ablation |
| Time/frequency masking | Simple regularization | Added noise/pitch/time changes need validation and can distort cues | No augmentation ablation |
| Class-weighted loss | Accounts for fewer neutral trials | Oversampling repeats examples; unweighted loss changes priorities | No ablation |
| Single actor split | Clear, affordable comparison | Group cross-validation measures variation more thoroughly | Cross-validation not run |
| Compact model from scratch | Transparent pipeline that fits the available GPU | Pretrained speech encoder may improve features; must verify pretraining data and avoid emotion-test overlap | Pretraining not tried |
| Validation macro F1 selection | Equal attention to each emotion | Accuracy emphasizes frequent classes; other metrics serve other objectives | Metric choice not optimized |

An **ablation** changes one component to test its contribution. Do not describe
an untested alternative as worse. A sound answer is: “This was a practical
baseline choice. I would test that alternative under the same actor split.”

## 11. Questions to rehearse aloud

**Why use a CNN for sound?** Its input is a numerical time-frequency matrix.
Convolutions can learn local patterns in that matrix. We are not pretending a
waveform is a photograph or using the Plotly colors as features.

**What is the difference between FFT and STFT?** One whole-clip FFT summarizes
frequency content; repeated windowed FFTs preserve when those components occur.

**Why log-mel rather than MFCC?** Log-mel retains a richer time-frequency grid for
the CNN. We also tested an MFCC-summary MLP; it had lower validation macro F1.
That compares whole pipelines, not the isolated effect of MFCCs alone.

**What does `model.eval()` do?** It changes dropout/normalization behavior.
`inference_mode()` separately turns off gradient tracking. Neither loads weights;
`load_state_dict()` does that.

**Why can validation accuracy exceed training accuracy?** Training batches are
masked, dropout is active and weights change throughout the epoch. Validation
uses fixed weights without those disturbances. Also, validation actors differ.

**What is overfitting?** Learning patterns specific to the training examples that
do not generalize. Our MLP's large training/validation gap illustrates it. Loss
weighting/dropout differences mean raw loss curves also require care.

**Why not just train longer?** The best validation checkpoint can occur before
the last epoch. More training may improve memorization without helping new voices.

**How reproducible is it?** Saved seeds, actor manifests, settings, checkpoints,
feature caches and an environment lock help. GPU nondeterminism and the
downloader's default “latest version” remain caveats; our dataset was version 1.

**Can it handle a microphone or a long conversation?** There is a WAV-file command,
not a microphone/streaming application. The selected model sees a centered segment;
real-time segmentation, silence handling and aggregation would need more work.

**Is 99.6% softmax confidence proof that the sample is calm?** No. The sample was
predicted calm, but scores are uncalibrated and its true label was not established
in that demo. A confident error is possible.

**What would you improve first?** Establish broader actor-group evaluation, then
compare targeted changes using validation only. A pretrained independent speech
encoder and better augmentation are possible experiments, not promised gains.

**Is the code production-ready?** It is a tested local training/evaluation baseline.
It lacks deployment monitoring, calibrated uncertainty, broad external validation,
streaming support and exact interrupted-training resume.

## 12. A short preparation route before Monday

**First pass, about 45 minutes:** read sections 1–4 and open the Fourier/audio
feature reports. Explain sample rate, waveform, spectrogram and model input
shape out loud. Avoid starting by memorizing library calls.

**Second pass, about 45 minutes:** read the comments in `feature`, `EmotionCNN`,
`train` and `score`. Point to each training-loop operation and describe its purpose.
If you cannot explain one, write that question down and revisit its paragraph.

**Third pass, about 30 minutes:** read sections 8–10 and `RESULTS.md`. Practice
explaining the actor split, the validation choice, 133/240 accuracy, and the sad
class failure. Know what was not tested.

**Final pass, about 30 minutes:** run the small demo below and answer five of the
questions without reading. Use the reports as evidence, not as a script to recite.
These are suggested study blocks; take more time wherever the concepts are new.

```powershell
# Run from this project folder. Loads the completed model; does not retrain it.
.\.venv\Scripts\python.exe ravdess.py predict --checkpoint runs/final/best.pt --audio sample_audio.wav --device cpu

# Opens no training job; regenerates the educational feature report.
.\.venv\Scripts\python.exe Audio_prep.py --audio sample_audio.wav
```

For a presentation, use the problem, input transformation, actor split, selected
model, result and limitation as your six talking points. Keep the interactive
reports available locally so the demo does not depend on internet access.

If asked about the six-year history, describe it plainly: you started with audio
exploration, paused while other commitments took priority, and recently returned
to complete the baseline with AI assistance. Distinguish what you originally
wrote, what was added recently, and what you can now explain and verify. You do
not need to imply six years of continuous development or audio expertise.

## Reading map

| File | What to learn there |
|---|---|
| `Audio_prep.py` | Waveform, frequency/time axes, FFT, STFT and MFCC plots |
| `Fourier_Transformation.py` | How two simple tones combine |
| `download_data.py` | Data acquisition and duplicate-folder handling |
| `analyze.py` | Data audit and descriptive plots |
| `ravdess.py` | Features, actor split, networks, training, evaluation and prediction |
| `finalize.py` | Validation-only selection followed by test evaluation |
| `reporting.py` | Reading curves, confusion matrices and prediction-score plots |
| `run_project.py` | How the experiments are reproduced in order |
| `test_ravdess.py` | What the implementation checks cover; these are not accuracy tests |
| `RESULTS.md` | Actual findings and limitations |
