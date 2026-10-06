# Source walkthrough and reading order

Read this beside the source, using [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) for
concepts and [README.md](README.md) for commands. Notes follow the statements in
each function in execution order. A multiline function call is one logical
statement; repeated chart/argument/assertion lines are explained together by
their shared role. The goal is to explain each line's **input, operation, output
and reason**, not memorize a paraphrase of Python syntax.

All 13 project-owned Python files are covered below. NumPy, librosa, PyTorch,
Transformers, Plotly and Gradio supply library implementations; this guide explains
the contracts our code uses rather than every line inside those dependencies.

## 0. What to read first

| Step | Source | Then read these notes |
|---:|---|---|
| 1 | [Fourier_Transformation.py](Fourier_Transformation.py) | Section 2 |
| 2 | [Audio_prep.py](Audio_prep.py) | Section 3 |
| 3 | [download_data.py](download_data.py) | Section 4 |
| 4 | [ravdess.py](ravdess.py): constants, records, split_records | Sections 5–6 |
| 5 | [analyze.py](analyze.py) | Section 7 |
| 6 | [ravdess.py](ravdess.py): feature, CNN, MLP | Sections 8–9 |
| 7 | [wavlm.py](wavlm.py) | Section 10 |
| 8 | [ravdess.py](ravdess.py): factory, restore, loader, score, train, CLI | Sections 11–15 |
| 9 | [finalize.py](finalize.py), [reporting.py](reporting.py) | Sections 16–17 |
| 10 | [run_project.py](run_project.py) | Section 18 |
| 11 | [app.py](app.py) | Sections 19–21 |
| 12 | [test_ravdess.py](test_ravdess.py), [test_wavlm.py](test_wavlm.py), [test_app.py](test_app.py) | Section 22 |
| 13 | Requirements, result documents, generated artifacts | Section 23 |

The source order in ravdess.py is convenient for definitions, while this study
order introduces data before the networks and networks before optimization.
You can temporarily skip an unfamiliar reporting call and return at section 17.

## 1. Repeated Python and tensor patterns

| Syntax / call | What it does here | Why it is present |
|---|---|---|
| Module docstring / `# comment` | Documents purpose / nearby choices | Study/help text; comments don't execute |
| `import x`, `from x import y` | Bind a module/function/class name | Use its API; module top-level code runs on first import |
| `as np/sf/gr/go` | Local alias for a module | Keep familiar API calls short |
| `def`, `class` | Define behavior; bodies execute on calls/instances | Reuse a clear contract |
| `if __name__ == '__main__'` | Call entry point only when directly executed | Import helpers without starting training/server |
| `Path(...)`, `/`, `.resolve()` | Portable path object, join, absolute path | File access and stable manifests |
| `parser.add_argument`, `parse_args` | Declare options, then read/validate CLI strings | Select data, model, limits and output |
| `type=int/float`, `choices`, `required`, `store_true` | Convert, constrain, require, or set a boolean flag | Avoid ad hoc argument parsing |
| `raise`, `try/except`, `raise ... from error` | Stop invalid work / translate expected errors / preserve cause | Make failures visible and understandable |
| `with` | Enter/exit a managed resource/context | Close files, remove temp folders, unlock, or disable autograd |
| `with gr.Blocks/Tabs/Tab` | Establish a UI construction context | Place components in the page hierarchy |
| `for`, `enumerate`, `zip` | Iterate values, values+index, aligned collections | Batches, rows, figures and matching labels/scores |
| List/set/dict comprehensions | Build a collection from an iteration | Features, label IDs, split mappings and chart series |
| `dict(...)`, `dict(CONFIG, kind=...)` | Create metadata or copy with added key | Avoid changing global preprocessing defaults |
| `**mapping`, `*sequence` | Unpack keyword / positional arguments | Processor tensors, model layers and runner paths |
| `self`, `super().__init__()` | Instance state; initialize base nn.Module | Register child layers, parameters and buffers correctly |
| `nn.Sequential` | Run layers in the listed order | A simple feed-forward network |
| `register_buffer` vs `parameters()` | Persistent nonoptimized tensor vs learned weights | Store train-fitted normalization separately |
| `[None]` | Add a size-one axis | Feature channel or one-recording batch |
| `[0]`, `[:, ...]`, `[-size:]` | Select a recording, slice axes, retain tail | Pool/infer/augment/bound a window |
| `axis=1` / `dim=1` / positional `1` | Reduce or operate along the second axis | Time statistics for MFCC; class softmax for logits |
| `np.stack` vs `np.concatenate` | Add a new axis vs join along an existing one | Batch features vs append waveform/feature components |
| `astype(np.float32)`, `torch.from_numpy` | Choose dtype; wrap array as a tensor | Model input compatibility; from_numpy shares CPU storage |
| `.to(device)`, `.cpu()`, `.numpy()`, `.tolist()` | Move tensor / convert representation | GPU arithmetic, NumPy pooling outputs, JSON/UI values |
| `.eval()`, `.train()` | Change layer behavior | Dropout/BatchNorm evaluation vs training |
| `inference_mode()` / `no_grad()` | Disable gradient tracking for a block | Faster/lighter forward-only work |
| `.state_dict()`, `load_state_dict` | Export/restore named parameters and buffers | Preserve learned network and preprocessing state |
| `json.dumps(..., indent=2)` | Serialize readable metadata | Auditability rather than a binary-only experiment |
| `flush=True`, `python -u` | Send printed progress promptly | See feature-loading/training/server progress |
| `lambda` | Small anonymous callback/key function | Selection keys and UI reset return values |

Import roles: standard-library modules handle arguments, files, JSON/CSV, hashes,
randomness, timing and processes. NumPy does array operations; soundfile reads/writes
WAV samples; librosa transforms/resamples audio; torch owns tensors/autograd/models;
sklearn computes metrics; Transformers loads WavLM; Plotly builds charts; Gradio
handles browser UI/events; imageio-ffmpeg locates the bundled decoder. Tests use
unittest/tempfile/Mock to isolate checks.

## 2. Fourier_Transformation.py: the simplest signal

Read `main()` in [Fourier_Transformation.py](Fourier_Transformation.py).

1. Imports expose numerical sine/FFT operations, Plotly traces and the common
   HTML writer. Importing this file defines main; the bottom guard runs it.
2. `sr = 1000` fixes the teaching sample rate, independent of model settings.
   `arange(sr) / sr` builds 1,000 timestamps spanning one second.
3. The two sine expressions use `2π f t` for 30 Hz and 70 Hz, multiplied by
   amplitudes 0.5 and 0.2. Addition combines signals sample by sample.
4. `go.Figure()` creates an empty chart. The tuple loop adds the two components
   and sum with labels. `tolist()` supplies serializable coordinates.
5. `update_layout` labels axes and zooms the view to 0.2 seconds; it does not
   slice the signal used by the FFT.
6. `rfft(first + second)` returns nonnegative-frequency complex coefficients
   of real samples. `abs` keeps magnitude; `rfftfreq(sr, 1/sr)` builds matching
   frequencies from signal length and sample interval.
7. Divide by signal length and multiply by two to recover amplitudes of these
   interior sinusoidal bins. A general estimator treats DC/Nyquist separately.
   Integer cycles fit the one-second window, so this example has clean peaks.
8. The spectrum layout limits the display to 100 Hz. `write_report` saves both
   charts to `runs/analysis/fourier.html`; no learning occurs.

**Explain:** Why is sample rate 1,000 while the peaks are 30 and 70? Measurements
per second and oscillations per second are different quantities.

## 3. Audio_prep.py: four views of one recording

Read `main()` in [Audio_prep.py](Audio_prep.py).

1. Declare audio/output arguments, parse them, and load mono audio resampled to
   16 kHz. The default input is sample_audio.wav; the default output is HTML.
2. Reject an empty/nonfinite waveform before plotting. This visualization does
   not explicitly reject silence as the prediction preprocessors do.
3. Waveform coordinates are sample indices divided by rate; values are amplitudes.
   Layout labels make the units explicit.
4. `abs(rfft(signal)) / len(signal)` plots whole-recording FFT magnitude,
   not power and not the factor-two single-sided amplitude used in the toy demo.
   It loses the times when frequencies occurred.
5. `rfftfreq(len(signal), 1/sr)` supplies frequency coordinates aligned to FFT
   values, rather than using arbitrary indices as Hertz.
6. `librosa.stft` uses 512-sample overlapping windows at hop 160. Taking its
   magnitude and `amplitude_to_db(..., ref=np.max)` creates a time/frequency
   magnitude view. There are 257 nonnegative FFT bins.
7. `frames_to_time` creates column times; `fft_frequencies` supplies row
   frequencies. Heatmap z is a matrix, not a color image used for training.
8. `mfcc(..., n_mfcc=20)` creates a teaching matrix. Its y axis is coefficient
   number, not Hertz. The same FFT/hop settings give matching frame times.
   This uses librosa's default mel count, while training explicitly uses 64.
9. `write_report` packages four (heading, description, figure) tuples. Path.name
   labels the input; length/rate labels duration. Printing resolve() gives the
   absolute output location.
10. The entry guard runs only on direct execution. This script has no trimming,
    cropping, model normalization, labels, loss or optimizer.

## 4. download_data.py: obtain one dataset copy

Read `main()` in [download_data.py](download_data.py).

1. Resolve `__file__` and take its parent to anchor cache/output paths to the
   project, regardless of where the shell started.
2. `os.environ.setdefault` sets a Kaggle cache default under data but respects
   any already configured path. Set it before importing kagglehub.
3. `dataset_download` fetches/reuses the dataset handle. The current handle
   requests latest; the completed experiment used version 1. The comment describes
   explicit version pinning for strict reproduction.
4. Construct a candidate nested directory and use it when present. Version 1
   contains two copies; choosing one avoids duplicated filenames/examples.
5. Create data if needed, save the selected path to dataset_path.txt in UTF-8,
   and print it. This file supplies other commands; it contains no labels/features.
6. The entry guard prevents a download merely from importing this module.

## 5. ravdess.py: constants and records()

Read [ravdess.py](ravdess.py) without starting at its training loop.

- `EMOTIONS` is ordered. List index 0–7 matches targets, output columns,
  classification report names and browser scores.
- `CONFIG` supplies rate 16,000, duration 4, FFT 512, hop 160 and 64 mel bands.
  A copied run config adds `kind` and WavLM metadata.
- `records(root)` initializes an output list and a set of seen filenames.
  Sorted recursive `*.wav` traversal makes row ordering stable.
- `re.fullmatch` requires audio-only speech and valid encoded fields/actors.
  A mismatch is skipped, so songs/unrelated files are excluded.
- A previously seen basename raises ValueError. This checks repeated folder
  copies; it does not compare waveform contents.
- Split the stem on hyphens and convert fields to integers. Emotion field 2 is
  converted from 1–8 to 0–7; actor field 6 is retained for grouping.
- Store an absolute path with label/actor. Nothing here loads audio, fits weights
  or exposes the filename label to the network.
- Fail when no matching records were found rather than training on nothing;
  otherwise return the metadata list.

## 6. ravdess.py: split_records()

1. Gather unique actor IDs with a set and sort them. Fewer than six is rejected
   because this procedure needs distinct train/validation/test groups.
2. Create `random.Random(seed)`, a separate random generator; shuffle the actor
   list without consuming the model's global random stream.
3. Compute `n = max(1, round(actor_count / 6))`. For 24, n=4. First n actors
   become test; next n validation; the remaining 16 training.
4. The dictionary comprehension retains rows whose actor belongs to each group.
   All recordings of one actor follow that actor.
5. For each partition, require the exact label set 0–7. This catches incomplete
   datasets where a reported eight-class evaluation would be misleading.
6. Return `{test, validation, train}` row lists. The insertion order can prepare
   validation features before training, but this does not fit normalization.

**Explain:** The actual split is 16/4/4 actors, not 80/10/10 files. Weight seeds
and split seeds are independent.

## 7. analyze.py: file audit and exploration

Read `main()` in [analyze.py](analyze.py).

1. Parse data/output/split-seed arguments and create the analysis output directory.
   Unlike experiment training, analysis allows an existing output.
2. Reuse records() and split_records(). The nested dict comprehension maps each
   actor to its single split. Initialize hash lookup, inventory and duplicate list.
3. For each row, soundfile reads original samples with `always_2d=True`, yielding
   `[samples, channels]` even for mono. Preserve original rate/duration/channels.
   Reject empty/nonfinite samples.
4. Hash the entire file's bytes with SHA-256. If a digest was seen, look up its
   previous row. Cross-split duplicates raise; within-split pairs are recorded
   and both named trials retained. Update the lookup.
5. Parse intensity/statement/repetition from filename fields. Add those plus
   metadata, duration `len/sr`, channel count, maximum absolute amplitude, RMS,
   near-full-scale sample fraction and digest to the inventory.
6. Progress prints every 240 files. It is operational feedback, not a model score.
7. Open CSV with UTF-8 and `newline=''`; DictWriter takes keys from the first
   inventory record, writes a header and all rows. This is metadata, not model input.
8. Build summary counts, unique rates/channels, duration bounds, total minutes,
   duplicates, zero-peak files, clipping flags, emotion counts and sorted actors
   per partition. Cross-split duplicates is zero only because any such case failed.
9. Save audit.json and splits.json using the shared JSON helper.
10. Create figures in order: stacked emotion counts per split; actor/emotion count
    heatmap; emotion duration boxplots; RMS versus duration scatter with filenames.
    List comprehensions select rows and counts; layout/hover labels expose units.
11. For the 4×2 log-mel example grid, choose the first training row for each emotion.
    Load/resample, compute 64-band mel power and peak-relative dB over the full
    original duration. No trimming, crop/pad or z-normalization is done for these
    illustrations. Time coordinates use the 10 ms hop. Grid location uses integer
    division/remainder of the emotion index.
12. Set axes/height, append the grid to the figure list, and call write_report
    with audit facts and scope. Print the absolute report path. Entry guard runs main.

**Explain:** File health/metadata checks can examine the dataset without training;
test prediction errors must not drive subsequent claimed untouched-test tuning.

## 8. ravdess.py: feature() in exact order

This extractor is for CNN/MFCC. WavLM dispatches to its own callable.

1. `librosa.load(..., sr=config['sr'], mono=True)` returns float samples at the
   chosen rate; underscore discards the returned rate because it was requested.
2. Reject empty/nonfinite samples and effectively silent peak amplitude.
3. Trim leading/trailing low-RMS frames using a 35 dB relative threshold. Underscore
   discards trim indices. This does not denoise or remove all pauses.
4. `config.get('kind') == 'mfcc'` selects the early-return branch:
   compute 40 MFCCs with FFT/hop from config and 64 mel filters; compute deltas
   with nearest edge handling; reduce each `[40,T]` matrix along time (axis 1).
   Concatenate mean, population std, delta mean and delta std into float32 `[160]`.
   It uses the whole trimmed clip.
5. Otherwise compute target sample count = rate × seconds. Center start is
   max(0, excess//2). Slice at most size samples and use fix_length to append zeros
   if short. This path has a fixed waveform length.
6. Compute mel power with saved FFT/hop/mel settings. Defaults produce `[64,401]`
   from a four-second waveform because librosa centers frames.
7. Convert power to peak-relative dB. Subtract the whole matrix mean and divide
   by its population std plus 1e-6. Epsilon avoids a zero denominator.
8. Cast to float32 and add a channel axis with [None]: `[1,64,401]`.
   DataLoader later adds batch, giving `[B,1,64,401]`.

**Explain:** The MFCC early return bypasses crop/padding. CNN normalization is
per clip, whereas MFCC input scaling is fitted in train() across training rows.

## 9. ravdess.py: the baseline networks

### EmotionCNN.__init__ and forward

1. Initialize nn.Module via super, an empty layer list and incoming channel count 1.
2. Loop over 16/32/64/128 output channels. Extend the list with Conv2d (3×3,
   padding 1), BatchNorm2d, ReLU, 2×2 MaxPool and 0.1 Dropout2d. Set incoming
   channels to the previous output for the next block.
3. Sequential unpacks all layers, then adds adaptive averaging to
   `(pool_bands,1)` and flatten. Selected CNN uses four frequency regions;
   global baseline uses one. Four blocks output `[B,128,4,25]` with defaults.
4. Classifier is 0.3 dropout followed by Linear(128 × pool_bands, 8).
5. forward() passes x through features, then classifier. It returns logits,
   not probabilities; gradients flow through this composition during training.

### EmotionMLP.__init__ and forward

1. Initialize nn.Module. Register mean zeros/std ones of length 160 as buffers.
   The initial values are placeholders overwritten by train-only estimates.
2. Build Linear(160,256), LayerNorm, ReLU, dropout 0.35; repeat for 256→128;
   finish with Linear(128,8).
3. forward() broadcasts `(x - mean) / std` over rows, then runs network.
   It returns `[B,8]`. Input buffers persist in the state dictionary.

## 10. wavlm.py: frozen encoder and trainable head

Read [wavlm.py](wavlm.py), then return to ravdess training.

### Constants and waveform()

- ENCODER_DIRECTORY fixes the sibling export folder name. DEFAULT_CHECKPOINT
  documents the model ID; the CLI currently repeats that default explicitly.
  FEATURE_VERSION is a manually maintained embedding implementation version.
- Require rate 16 kHz and positive crop duration. Load at source rate with
  channels preserved: librosa stereo is `[channels, time]`.
- Reject empty, nonfinite or silent values before conversion. Downmix to mono,
  resample, trim edge silence, compute crop size/start and slice centered audio.
- Cast float32; do not call fix_length, so short clips remain short. Reject fewer
  than 400 samples for base-plus's convolutional receptive field; return waveform.

### FrozenWavLM.__init__()

1. Lazy-import Transformers/config/processor/model. If missing, wrap ImportError
   with the project installation instruction.
2. Store config by reference and create a torch.device. Updating config here also
   updates the dictionary later written by training and hashed by loader.
3. **Local branch:** require the encoder directory, convert its Path to string
   and set local_files_only. Read architecture from those files.
4. **Hub branch:** take model ID and requested revision, set download cache, read
   AutoConfig and resolve its commit hash. Reject an unresolved revision.
   Replace revision in loading options with the commit and add version, pooling,
   precision and library metadata to config.
5. Verify architecture type is wavlm, not merely a suggestive repository name.
   Load the audio processor and require its sample rate to equal ours.
6. Load WavLMModel with that architecture/options. Move device, set evaluation
   mode, and disable gradients for all encoder parameters.
7. Compute embedding size = 2 × hidden_size. When restoring locally, verify
   it matches the saved head's expectation. Write embedding_dim into config.

### save() and __call__()

- save() exports encoder tensors with safetensors plus architecture config and
  audio processor settings. It does not write emotion-head weights.
- __call__() requires matching config, obtains waveform, and calls the saved
  processor with padding=False and PyTorch tensor output. Each returned tensor
  (input_values and any attention_mask) moves to the encoder device.
- inference_mode disables graph creation. Unpack tensor keywords into encoder,
  select recording 0 from last_hidden_state and cast float32. Base-plus shape
  is `[1,T,768] → [T,768]`.
- Compute time-wise mean and population std (unbiased=False), each `[768]`;
  concatenate to `[1536]`. No padded samples enter these statistics.
- Move to CPU, convert to NumPy and copy independent storage. Reject nonfinite
  pooled values; return one feature vector.

### WavLMClassifier.__init__ and forward

Initialize nn.Module, register embedding-sized mean/std buffers, then build
embedding→256 Linear, LayerNorm, ReLU, dropout 0.35 and Linear→8. forward standardizes
each embedding using saved buffers and returns eight logits. Only this head
is passed to the training optimizer; the encoder stays separate and frozen.

**Explain:** This is a frozen pretrained feature extractor. It does not learn the
emotion classes until our head trains, and it is not encoder fine-tuning.

## 11. ravdess.py: model construction, restoration and device

### make_model()

- For wavlm, lazy-import WavLMClassifier and use saved embedding_dim; no encoder
  is built by this factory.
- For mfcc, instantiate EmotionMLP. Reject unknown kinds.
- For cnn, instantiate EmotionCNN with pooling count. This creates weights/layers;
  loading learned values is a separate operation.

### load_checkpoint()

1. torch.load reads inference metadata/state with weights_only=True and
   map_location so GPU-saved tensors can be restored on CPU.
2. Get saved config/model kind; older checkpoints default to CNN. Construct
   compatible dimensions/pooling, move device, then load the state dictionary.
3. eval() changes layer behavior for prediction. It does not by itself disable
   gradients, so forward callers also use inference_mode.
4. Start with feature as extractor. For WavLM, instantiate FrozenWavLM using
   `checkpoint parent / wavlm_encoder`, requiring local files.
5. Return checkpoint, model and callable extractor, a shared contract for CLI,
   finalizer and web app.

### device_for()

Explicit unavailable CUDA raises a clear error. Auto selects CUDA if available,
otherwise CPU; an explicit supported name becomes that torch.device. GPU placement
changes compute resources, not target labels or statistical evidence.

## 12. ravdess.py: loader() and feature cache

1. Require a loaded extractor for WavLM, even when a cached vector might exist.
   Otherwise default to feature().
2. For each metadata row, create a file Path; serialize absolute path, size,
   modification time in nanoseconds and full config to JSON.
3. Hash that metadata string with SHA-256 for a filesystem-safe cache basename.
   This is not an audio-content hash or automatic source-code versioning.
4. If cache is provided and its .npy file exists, load with allow_pickle=False.
   Otherwise call extractor, create cache parent if needed and save the array.
5. For WavLM verify shape/finite values and print periodic extraction progress.
   Append each feature in original row order.
6. Stack vectors/matrices into a batchable CPU tensor:
   `[N,1,64,401]`, `[N,160]` or `[N,1536]`.
   Build class targets as a long tensor, suitable for cross entropy.
7. TensorDataset pairs inputs/labels. DataLoader returns minibatches of requested
   size, optionally shuffled; zero workers simplifies Windows loading.
   Train uses shuffle; evaluation retains row alignment.

All features live in CPU RAM here, not lazy on-disk batch extraction. WavLM
embeddings can be reused across epochs only because its encoder is frozen.
Changing CNN/MFCC extraction requires rebuilding caches; WavLM config includes
a feature version that must be bumped deliberately when semantics change.

## 13. ravdess.py: score(), save_json() and plot_results()

### score()

1. Set eval mode, create truth/prediction/probability lists and zero loss sum.
2. Inside inference_mode, move each x to device and compute logits. Move y for
   unweighted cross entropy with reduction='sum', accumulating recording losses.
3. Class-axis argmax gives predicted IDs. Softmax gives eight scores; move to CPU
   and Python lists for reporting. Original y contributes truth in loader order.
4. Divide total loss by recording count. Compute accuracy and macro F1 over all
   eight labels, a classification report with emotion names, and an 8×8 confusion
   matrix converted to a list. zero_division=0 handles undefined class metrics.
5. Return all metrics plus per-recording arrays. No backward or optimizer step.

### Helpers

save_json converts a serializable value to indented JSON and writes UTF-8; its
caller creates the directory. plot_results lazily imports training_report and
passes history, metrics and the split name to a fixed report.html destination.

## 14. ravdess.py: train() line groups in execution order

### Setup and preparation

1. Reject nonpositive epochs, batch size or patience. Set global Python/NumPy/
   torch seeds for initialization, shuffle and augmentation repeatability.
2. Resolve device, copy defaults with model kind and create a **new** output
   directory (exist_ok=False). Fail rather than overwrite an existing experiment.
3. Parse/split rows using split seed and save splits.json.
4. Set extractor to feature. WavLM branch adds requested Hub settings, constructs
   FrozenWavLM (resolving metadata/dimensions) and exports its local encoder.
5. Save arguments (`vars(args)`), preprocessing, torch version and device to
   config.json. Print feature preparation progress.
6. The batches comprehension excludes test and prepares validation/train,
   passing cache and extractor. Only the train loader shuffles.
7. Construct the trainable model on device. MFCC/WavLM branches access the train
   dataset tensor, fit mean/std across rows, and copy into buffers. Clamp std
   to at least 1e-5. torch's default across-recording std uses sample correction;
   within-clip NumPy/std pooling above uses population std.

### Objective and optimizer

8. bincount gets all eight train class counts. Make float32 weights
   `N/(8*count)` on device for CrossEntropyLoss; neutral has twice the relative
   weight of another class.
9. AdamW receives only model.parameters(), LR from CLI and weight decay 0.01.
   For WavLM, model is the head, not the frozen encoder.
10. ReduceLROnPlateau maximizes validation macro F1, halves LR on a plateau and
    has patience 5. GradScaler is enabled only on CUDA.
11. Initialize history=[], best=-1 and stale=0. First finite F1 can beat -1.

### Epoch and minibatch loop

12. range(1, epochs+1) gives human-readable epochs. monotonic() starts elapsed
    timing; model.train() enables training behavior. Reset loss/correct counters.
13. Move x/y for each training minibatch. CNN-only branch chooses random valid
    mask starts and writes zeros over six mel bands and 20 frames.
    These are batch tensors, not a persistent rewrite of cached feature files.
14. zero_grad(set_to_none=True) clears previous gradients efficiently.
15. autocast on CUDA surrounds forward/loss. Network returns logits; criterion
    combines logits with class IDs into one differentiable scalar.
16. Scaled loss.backward() computes gradients. unscale_ restores actual gradient
    scale; clip_grad_norm_ caps global norm at 5. scaler.step calls optimizer
    if gradients are valid; scaler.update adjusts the numerical scale.
17. Accumulate loss.item() × batch length and argmax matches. item() converts a
    scalar tensor to a Python value; it is for logging, not differentiable learning.

### Validation, selection and completion

18. score() switches to eval/inference mode. Append epoch train/validation
    statistics, current LR and elapsed seconds to history. LR is recorded before
    the following scheduler step; seconds excludes initial feature extraction.
19. scheduler.step(validation F1) may change LR. Print latest row and save history.
20. Strict F1 improvement sets best and resets stale; torch.save writes model state,
    preprocessing/labels/kind, chosen epoch, weight seed, pooling and split seed.
    Otherwise increment stale. Exact equal F1 is not a new best epoch.
21. Break when stale reaches patience, even if maximum epochs wasn't reached.
22. Reload best.pt state rather than using the final epoch. Score/save best
    validation metrics. This checkpoint lacks optimizer/RNG resume state.
23. If skip_test is false, now prepare/score test and save test_metrics.json.
    For candidate comparison always use skip_test and finalize separately.
24. Plot validation results when test was skipped, otherwise test results, alongside
    the learning history. No extra training is done at reporting time.

**Explain:** Training loss is an aggregate of weighted minibatch means;
validation loss is unweighted per-recording CE. Their values have different
weighting. Higher validation than training accuracy can arise from disabled
dropout/masking, not just leakage.

## 15. ravdess.py: main() CLI and prediction

1. Build argparse with required subcommand. Train options cover data/output,
   maximum epochs, batch/patience/LR, model/split seeds, model type, WavLM source/
   revision, cache, skip-test and CNN pooling count.
2. A loop builds predict/evaluate parsers: both require checkpoint; predict
   requires audio, evaluate manifest; optional output writes files.
   Another loop adds auto/cuda/cpu to all commands. Parse once.
3. Train dispatches train(args) and returns so it doesn't enter inference code.
4. Otherwise resolve device and restore checkpoint/model/extractor.
5. Evaluate reads the manifest's **test** list, loads features in order, scores,
   optionally creates output/JSON/HTML and prints summary metrics without the
   potentially large truth/prediction/probability arrays.
6. Predict uses the saved config with its extractor. from_numpy makes a tensor,
   [None] adds batch and to moves device. Shapes are `[1,1,64,401]`, `[1,160]`
   or `[1,1536]`.
7. inference_mode runs model then class softmax. [0] selects the one recording;
   cpu().tolist() prepares reporting values.
8. np.argmax selects index and EMOTIONS names it. zip pairs all names/scores
   into a dictionary. Optional output writes prediction.json and an offline bar
   chart; json.dumps prints the same result.
9. The guard runs main only for direct execution. Importing functions never parses
   CLI arguments or starts training.

The CLI takes local audio paths; unlike Gradio, it doesn't run the bundled
browser-container decoder or enforce the web upload size/duration limits.

## 16. finalize.py: freeze the decision, then score test

Read `main()` in [finalize.py](finalize.py).

1. Parse one-or-more run folders, new output and device. Create output with
   exist_ok=False. Initialize candidates and reference manifest.
2. For each run, load splits.json and require exact equality to the reference,
   including paths/order. Load validation metrics and retain run/accuracy/F1.
3. max(..., key=validation_macro_f1) chooses the winner; first exact tie wins.
   Save selection.json before making test predictions.
4. Copy best.pt, splits, config, history and validation metrics. If the source has
   wavlm_encoder, copy it recursively too. No train+validation retraining occurs.
5. Restore with the shared factory/extractor; read reference test rows.
   loader uses batch 16, no shuffle and cached features; score returns aligned
   per-recording outcomes.
6. Build NumPy actor/truth/prediction arrays. For each actor create a boolean mask,
   recording count, accuracy and eight-class macro F1. Convert types for JSON.
7. Calculate the largest class fraction as the majority baseline and add per-actor
   metrics. Save test_metrics.json.
8. CSV writer records file/actor/true name/predicted name/correctness and all eight
   scores. zip preserves alignment guaranteed by ordered evaluation.
9. Read copied history and render winner's test report. Build validation-only
   candidate comparison and winner-only per-actor test bar charts.
10. write_report creates comparison.html; printed JSON exposes selected candidate
    and test outcome. Its narration assumes the project's normal 24-actor split.

The function selects among **provided candidates**, so its winner depends on
the list. The original four-run final remains CNN; the later five-run final is
WavLM. Reading their saved selection records resolves which experiment is meant.

## 17. reporting.py: metrics become offline HTML

Read [reporting.py](reporting.py). This module does no inference or training.
Its EMOTIONS constant mirrors the source label order without importing ravdess
and making reporting depend cyclically on the training module.

### write_report()

1. Accept destination, title, introduction and figure tuples.
2. For each figure, set template/font/margins/default height; preserve an explicit
   height such as the dataset grid's. to_html returns a fragment.
3. Embed Plotly JavaScript only for figure index 0, so subsequent charts reuse
   the library in the same offline document. Configure responsive charts and
   predictable div IDs.
4. Escape headings/descriptions/title/introduction as ordinary HTML text.
   Append chart fragments within section elements.
5. Combine document structure, CSS and sections. The CSS controls presentation,
   not model features. Create parent directory and write UTF-8 HTML.

### training_report()

1. If history exists, make a two-panel curve plot, gather epoch coordinates and
   add train/validation loss plus accuracy/F1 traces. The labels distinguish
   weighted and unweighted losses; set axes/legend and add to figures.
2. Convert confusion counts to NumPy. Divide each row by its total, guarding zero
   with max(...,1); diagonal fractions are class recalls.
3. Loop over raw and normalized matrices, formatting counts versus percentages;
   true classes are y, predictions x. Reverse y to keep first emotion at top.
4. Build grouped precision/recall/F1 bars by reading each emotion's report entry.
5. Convert per-recording lists to arrays. Maximum score along class axis gives
   the top softmax score; truth==prediction and its complement separate correct/
   incorrect score histograms. This is not a calibration curve.
6. Compose accuracy/F1/recording count text and call write_report.

### prediction_report()

Build one bar per emotion score, fix score-axis bounds to 0–1, look up a
model-specific description of the analyzed audio, and write_report with the
predicted label/input basename. These descriptions distinguish full MFCC
statistics from CNN/WavLM centered segments.

## 18. run_project.py: orchestration, not a second training implementation

Read [run_project.py](run_project.py).

- run(*arguments) uses sys.executable (the caller's Python environment), stringifies
  paths/values and executes a list of arguments with check=True. No shell parsing;
  a failed step stops the pipeline.
- main parses data/output/device and include-wavlm. Refuse an existing output
  before starting work.
- If data was provided, use it; otherwise run downloader then read/strip its
  saved path. Call analyze.py first.
- The specs list contains tuples of run name, model, weight seed, CNN bands,
  maximum epochs, patience and batch. Four original runs compare CNN global/
  bands/second seed and MFCC. Optional flag appends WavLM.
- For each tuple, create target path and call ravdess train with skip-test.
  pool-bands is passed uniformly but only affects CNN. Split seed remains 42
  because the CLI default is unchanged. Keep targets for finalization.
- After all sequential runs, invoke finalize once with the candidate list.
  This is reproducible orchestration, not guaranteed identical GPU arithmetic.

Run from project root: subprocess filenames and default dataset-path lookup are
relative to the shell's working directory. The runner does not start the web app.

## 19. app.py: constants, decoding and buffers

Read [app.py](app.py) after you understand checkpoint restoration.

### Initialization and decode_upload()

1. Standard imports support CLI, environment, file paths, decoder subprocess,
   temporary files and a lock. ROOT anchors model/cache paths to the script.
2. Set Gradio cache/analytics defaults **before** importing it. setdefault respects
   explicit environment settings. Third-party imports expose UI/audio/tensor APIs.
3. Constants fix mono rate 16 kHz, window four seconds and upload duration 30.
   MODEL_PATHS maps visible choices to trained checkpoints; accepted suffixes
   allow supported browser/mobile audio containers.
4. decode_upload converts its input to Path and rejects nonexistent/unsupported
   files or bytes exceeding 25×1024².
5. Locate bundled FFmpeg and run argument list without a shell. Disable interactive
   stdin; use error-only output; restrict file/pipe protocols and demuxer formats.
   Decode up to 31 seconds, output mono 16 kHz little-endian float32 to stdout.
   capture_output collects bytes; timeout=20 bounds decoding.
6. A nonzero return code becomes a friendly failure. frombuffer(...,'<f4')
   interprets bytes as samples; copy gives independent writable storage.
7. Reject decoded length >30×16000, then validate samples. The extra decoded
   second detects overlength instead of accepting a silently cropped upload.

### validate_samples(), microphone_samples(), append_window()

- validate_samples casts float32, requires one dimension, nonempty length and
  finite values, then returns the array.
- microphone_samples unpacks (rate,samples), requires an integer rate 8–192 kHz
  and one/two dimensions with a bounded chunk length.
- Signed PCM divides by the negative minimum's magnitude; uint8 subtracts
  midpoint 128 and scales by 128. Other arrays cast float32 without PCM scaling.
- For `[samples,channels]`, reject >8 or zero channels and average channels.
  Validate resulting mono samples, then resample to 16 kHz.
- append_window converts the pair, substitutes an empty array for None state,
  concatenates prior/new audio and retains the last 64,000 samples.
  Memory stays bounded; it does not retain an entire conversation.
- The actual live gr.Audio uses **filepath**, decoded first and wrapped as
  (16000,samples); the helper also supports arrays tested independently.

## 20. app.py: EmotionService and UI callbacks

### EmotionService.__init__ and predict()

1. Resolve selected device, initialize shared model mapping and threading.Lock.
   Optional paths allow a different checkpoint mapping.
2. Require each checkpoint file, print loading progress and restore through the
   common load_checkpoint contract. Require matching emotion label order.
3. Store (checkpoint, model, extractor) once per choice, avoiding loading weights
   for every request.
4. predict validates model name and waveform, minimum quarter second and a
   nonzero peak. The four-second live-context rule is enforced by its callback,
   not by this generic inference method.
5. Enter shared lock and TemporaryDirectory. Write float WAV at 16 kHz, preserving
   decoded samples through the path-based training extractor contract.
6. Call matching extractor with saved config; wrap tensor, add batch and place
   on selected device. inference_mode runs network + softmax.
7. Return name→score mapping. Context exit deletes our intermediate WAV and
   releases the lock even on error. Gradio's source files have separate cleanup.

### build_app(): upload_predict()

Require an uploaded path; decode it and call service.predict. Expected ValueError
or decoder timeout becomes gr.Error in the browser. Return scores/status matching
the two declared component outputs.

### build_app(): stream_predict()

1. If chunk is None, gr.skip preserves displayed result while returning unchanged
   buffer and waiting status.
2. Decode filepath, create (rate,samples), append to session window.
3. Before four seconds have accumulated, return empty scores/new buffer/progress.
4. Calculate RMS; very quiet windows return empty scores with advice, retaining
   buffer. This fixed threshold is not learned speech detection.
5. Otherwise infer on that window and return scores/buffer/status.
   Expected errors clear scores/reset buffer to None with the error text.

Callbacks are nested so they close over service. Their output tuple order must
match the [result, buffer, status] component list.

## 21. app.py: components, events, queue and main()

### build_app() page and events

1. Blocks declares title and hourly deletion checks for Gradio cached files older
   than an hour. Markdown supplies page instructions.
2. Dropdown lists loaded models, preferring WavLM as initial choice. Tabs create
   upload file/button and a microphone component.
3. File input supplies a filepath; Audio uses microphone source, filepath type,
   no forced Gradio format conversion and streaming=True. gr.State starts None
   with a one-hour TTL and belongs to each session.
4. Shared result label shows up to eight scores; status is read-only. Markdown
   states training domain and score meaning.
5. Button click connects [upload,model] to [result,status]; exposes predict_clip
   API with inference concurrency group/limit one.
6. mic.stream connects [mic,model,buffer] to [result,buffer,status].
   stream_every=1 requests nominal one-second chunks; time_limit=30 is a
   scheduling budget, not microphone auto-stop. Stream concurrency is four and
   API visibility private. These limits don't override the shared inference lock.
7. Start lambda clears buffer/result and sets listening text. Clear lambda resets
   them to ready and cancels the stream. Stop lambda clears buffer/status and
   cancels the stream, preserving last scores. queue=False makes reset actions
   avoid the regular queue; already executing work cannot be undone.
8. Model-change lambda clears result/status without clearing buffer; next chunk
   uses the selected model. Thus the same recent audio can feed a new model.
9. demo.queue configures max 16 pending events and default concurrency one.
   Return the constructed app; construction alone does not launch a server.

### main()

1. Parse auto/cpu/cuda, share flag and explicit port default 7860.
2. Instantiate service and generate a four-second 220 Hz tone at amplitude 0.1.
   Predict once per model to exercise preprocessing/device kernels. This isn't
   an emotion accuracy test.
3. Build app, then launch loopback server on selected port. share=True creates a
   public HTTPS relay; loopback itself remains local to the laptop.
4. Limit Gradio upload size, suppress detailed browser errors, and block serving
   model/repository/environment folders. This isn't a production security audit.
5. Direct execution guard calls main; importing app exposes helpers but does not
   load all models or start the server.

**Explain:** State is per visitor; weights and lock are shared. The app performs
repeated independent window inference, with no optimizer, log, ASR, result smoothing
or guaranteed wall-clock latency. Transport/queue backlog can lag current speech.

## 22. All three test files: what each assertion checks

Tests use unittest.TestCase methods named test_* for discovery. TemporaryDirectory
isolates files and cleans them afterward. assertEqual/True/False check invariants;
assertRaises verifies failure contracts; assert_close/allclose check numerical
consistency. Generated tones have no emotion ground truth. The bottom guards
allow direct execution; `python -m unittest -v` discovers all 13 tests.

### test_ravdess.py: PipelineTests (six)

| Method | Setup, operations and assertions |
|---|---|
| test_actor_splits_are_disjoint_and_repeatable | Create metadata for 24 actors × eight labels; split twice with seed 42; compare entire results; check actor counts 16/4/4, pairwise set intersections empty and no row loss. |
| test_parser_filters_song_and_rejects_duplicates | Touch speech/song filename placeholders; parse without decoding; check only speech included with zero-based label/actor; create second folder with same basename and expect error. |
| test_audio_shapes_and_checkpoint_roundtrip | Write 0.2- and 5-second stereo tones at 22,050 Hz; extract CNN features; assert equal shapes/finite values; stack batch; compute eval logits; save/reload state_dict into same CNN architecture; compare outputs with gradients disabled. |
| test_cuda_mixed_precision_training_step | Skip unless CUDA exists; create CNN/AdamW/scaler; run CUDA random batch with autocast and labels, backward/step/update; require finite loss. This exercises real GPU operations. |
| test_mfcc_and_saved_normalization | Generate tone; require MFCC batch shape (1,160); set known mean/std buffers; save/restore state and compare logits, verifying input scaling persists. |
| test_silent_audio_rejected | Write all-zero samples and require a ValueError mentioning silence from feature(). |

The round-trip CNN in tests uses the class's global-pooling default, while the
selected training factory uses four bands. Identical architecture on restoration
is the point of that test, not recreating the best trained checkpoint.

### test_wavlm.py: WavLMTests (three)

| Method | Setup, operations and assertions |
|---|---|
| test_frozen_encoder_and_offline_prediction_roundtrip | Save a tiny random WavLM (hidden=8, one transformer layer) and local 16 kHz processor. Require eval/frozen encoder; write short stereo tone; check unpadded resampling to 3,200 samples and reproducible 16-element mean/std embedding. Build head, set buffers, backprop/update it; require no encoder gradients and finite loss. Save head beside encoder, restore offline, compare logits and (1,8) shape. |
| test_cache_reuses_features_and_invalidates_on_revision | Touch input path; Mock extractor returns a 16-element array. Two identical loader calls invoke extraction once; changed revision invokes again. Missing frozen extractor must raise, even with a cached file. |
| test_crop_and_invalid_audio | Write 80,000 float samples and require center slice [8000:72000]. Write silence, only 100 samples and NaNs; require appropriate errors. |

Tiny dimensions reduce cost and avoid the internet. Head .eval() still allows
backpropagation; it disables dropout. Tiny random weights check APIs, not
pretrained emotion accuracy.

### test_app.py: WebAudioTests (four)

| Method | Setup, operations and assertions |
|---|---|
| test_live_callback_waits_for_context_and_resets_between_recordings | Mock service score; build UI, find stream_predict callback; repeatedly pass one-second WAV. First three calls wait without inference; fourth predicts and holds 64,000 samples; fifth predicts with unchanged size. None initial state starts a separate recording and waits again. |
| test_resampling_integer_stereo_and_bounded_session_buffers | Feed one-second 48 kHz stereo int16 constant, check resampling to 16 kHz and amplitude ~0.5 away from filter edges. Seven appends stay four seconds. A separately initialized zero buffer remains independent. |
| test_decode_wav_and_browser_webm | Write one-second 24 kHz tone; decode to 16 kHz finite samples. Use bundled FFmpeg to encode Opus WebM, decode it, allow codec-frame length tolerance (<320 samples), and check nontrivial amplitude. |
| test_empty_invalid_and_long_uploads | Reject empty/NaN samples, invalid rate and excessive channels; write 31-second WAV and require the 30-second upload error. |

These tests call Python callbacks directly. No browser microphone permission,
actual recording hardware, public relay, service fairness or exact one-second
wall-clock output guarantee is verified by them.

## 23. Files outside the Python source

| File / directory | How to read it / why it exists |
|---|---|
| [README.md](README.md) | Operational entry point: install, start/check demo, train, predict, evaluate and artifact map |
| [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) | Explain reasoning, alternatives, results and limits in your own words |
| [RESULTS.md](RESULTS.md) | Historical four-run CNN/MFCC experiment, not the current WavLM winner |
| [WAVLM_RESULTS.md](WAVLM_RESULTS.md) | Later frozen-encoder result on the same split and its saved settings |
| [requirements.txt](requirements.txt) | Direct package dependencies and compatibility bounds; imports link code to packages |
| [requirements-lock.txt](requirements-lock.txt) | Exact installed packages for the verified environment, including transitive dependencies |
| [.gitignore](.gitignore) | Excludes local environment, caches, data, model weights and generated run artifacts |
| `.venv/` | Installed Python environment, not project-owned model code |
| `data/dataset_path.txt`, dataset WAVs | Dataset location and samples; filenames encode labels for supervised experiments |
| `data/features/*.npy`, `data/huggingface/` | Cached arrays and downloaded encoder assets |
| `data/gradio_cache/`, `.gradio/` | Runtime files created by Gradio; not a new model or training corpus |
| `runs/*/config.json` | CLI settings plus actual preprocessing/version/device metadata |
| `runs/*/splits.json` | Ordered records and actor assignment; paths are absolute |
| `runs/*/history.json` | Read epoch, losses, accuracy, validation F1, LR, seconds to trace optimization |
| `runs/*/best.pt`, `wavlm_encoder/` | Binary model state and local encoder files; load through code rather than read as prose |
| `selection.json`, metrics JSON, predictions CSV | Evidence of candidate choice and actual per-recording/per-actor outcomes |
| Generated HTML | Read charts and hover details; rendering code is reporting.py |
| `sample_audio.wav` | Unlabeled functionality example, not a benchmark with known emotion |

Train-fitted values are learned state; source hyperparameters are choices; cached
arrays are derived data. Keeping those categories distinct makes it easier to
explain how the same source serves different trained checkpoints.

## 24. Trace a complete example yourself

For a training file, narrate: records label/actor → actor split → correct extractor
→ cache/batch → model logits → loss/backward/optimizer → validation score →
best checkpoint. For final evaluation: matching manifests → validation winner →
saved selection → fixed winner's test outcomes. For a microphone chunk: browser
filepath → decoder → per-session window → saved extractor/head → eight scores.

At each arrow, name the type/shape and whether anything is learned. If you can
do that for CNN, MFCC and WavLM, then explain the exceptions/error checks and
the test evidence, you can account for the project's code rather than just its
high-level architecture.
