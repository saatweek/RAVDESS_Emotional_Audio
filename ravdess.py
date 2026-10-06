"""Shared CNN/MFCC/WavLM training, evaluation, checkpoint and WAV prediction code.

Study route: records -> split_records -> feature -> networks -> loader -> score
-> train -> load_checkpoint -> main. wavlm.py supplies the pretrained branch;
app.py reuses load_checkpoint for web inference. See CODE_WALKTHROUGH.md.
"""
# Standard-library imports: CLI parsing, readable run metadata, repeatable random
# choices, filename validation, cache fingerprints, timing and portable paths.
import argparse
import json
import random
import re
import hashlib
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# librosa/NumPy create audio features on CPU; PyTorch holds models/tensors and
# trains them; sklearn computes evaluation metrics, not model predictions.

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# Read INTERVIEW_GUIDE.md and CODE_WALKTHROUGH.md first. List positions define
# target IDs and output order in all three models, reports and the Gradio UI.
# sr: samples/second; seconds: CNN input duration; n_fft: analysis window size;
# hop_length: samples between windows; n_mels: number of frequency summaries.
# At 16 kHz, these settings mean a 32 ms window and a 10 ms hop. They are
# reasonable baseline choices, not values we proved optimal in a parameter search.
CONFIG = dict(sr=16000, seconds=4, n_fft=512, hop_length=160, n_mels=64)


def feature(path, config):
    """Return CNN [1,64,401] or MFCC [160] features, using saved settings.

    WavLM has its own callable extractor; callers must dispatch to that extractor.
    The CNN/MFCC computations here are deterministic for a given file/settings.
    """
    # STEP 1: A WAV becomes a sequence of amplitude measurements, not text.
    # Resample every recording to the same rate so the same frequency has the
    # same meaning for every input. Mono removes the extra stereo dimension.
    # Keeping 48 kHz/stereo could preserve more detail but costs more processing;
    # we chose 16 kHz mono for a compact speech baseline, not as a proven optimum.
    audio, _ = librosa.load(path, sr=config['sr'], mono=True)
    # Invalid/silent input should fail visibly instead of producing an arbitrary
    # emotion. A silence detector is a separate concern from emotion recognition.
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Empty or invalid audio: {path}')
    if np.max(np.abs(audio)) < 1e-8:
        raise ValueError(f'Audio is silent: {path}')
    # Remove quiet leading/trailing frames relative to this clip's peak RMS.
    # top_db=35 is a relative dB threshold, not 35 seconds or an absolute loudness.
    # This is NOT denoising or a learned voice-activity detector; interior pauses
    # remain. Trimming can also discard useful quiet speech, a real tradeoff.
    audio, _ = librosa.effects.trim(audio, top_db=35)
    if config.get('kind') == 'mfcc':
        # ALTERNATIVE REPRESENTATION: MFCCs compress the log-mel spectral envelope
        # using a cosine transform. Their axes are coefficients, not frequencies.
        # Delta coefficients approximate how those values change across frames.
        mfcc = librosa.feature.mfcc(y=audio, sr=config['sr'], n_mfcc=40,
                                    n_fft=config['n_fft'], hop_length=config['hop_length'], n_mels=64)
        delta = librosa.feature.delta(mfcc, mode='nearest')
        # Average/spread over time turns variable-length audio into 160 numbers:
        # 40 coefficients x (mean, std, delta mean, delta std). This is cheap but
        # discards temporal order. The MLP uses the full trimmed clip, not a crop.
        return np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1)]).astype(np.float32)
    # STEP 2: The CNN needs stackable inputs. Four seconds at 16 kHz = 64,000
    # samples. Center-crop long clips and append zeros to short ones. Alternatives
    # are random training crops, multiple-window inference, or variable-length
    # batches with masks; those preserve/augment time differently but add logic.
    size = config['sr'] * config['seconds']
    start = max(0, (len(audio) - size) // 2)
    audio = librosa.util.fix_length(audio[start:start + size], size=size)
    # STEP 3: Short overlapping windows reveal how frequencies change over time.
    # librosa computes squared STFT magnitudes (power) and sums them through 64
    # overlapping mel filters. The mel scale emphasizes lower-frequency detail.
    # We feed these numbers directly to the CNN, not a colored screenshot.
    mel = librosa.feature.melspectrogram(y=audio, sr=config['sr'], n_fft=config['n_fft'],
                                         hop_length=config['hop_length'], n_mels=config['n_mels'])
    # Log compression reduces the huge range of power values. Relative to the
    # clip maximum, 0 dB is its strongest mel cell, not a universal sound level.
    mel = librosa.power_to_db(mel, ref=np.max)
    # STEP 4: Give each clip roughly zero mean and unit standard deviation.
    # This reduces recording-scale variation but also removes absolute loudness
    # cues that could help classify acted intensity. No dataset-wide statistics
    # are fitted here, so the operation can also be applied to one new clip.
    # epsilon avoids division by zero; float32 is the usual neural-network dtype.
    # [None] adds the channel axis: [1, 64, 401] with librosa's centered framing.
    return ((mel - mel.mean()) / (mel.std() + 1e-6)).astype(np.float32)[None]


def records(root):
    """Return metadata dictionaries; do not decode audio or learn features."""
    # Ground-truth emotion and actor come from RAVDESS filenames. They are used
    # for supervision/splitting, NEVER supplied to the model as input features.
    result = []
    seen = set()
    for path in sorted(Path(root).rglob('*.wav')):
        # 03 = audio only; 01 = speech. Ignore songs and unrelated WAV names.
        if not re.fullmatch(r'03-01-0[1-8]-0[12]-0[12]-0[12]-(0[1-9]|1[0-9]|2[0-4])', path.stem):
            continue
        if path.name in seen:
            raise ValueError(f'Duplicate recording: {path.name}')
        seen.add(path.name)
        # Filename duplication detects repeated folder copies. analyze.py also
        # checks file hashes, which can catch identical bytes with different names.
        parts = list(map(int, path.stem.split('-')))
        # Dataset emotions are 1..8; PyTorch class indices must be 0..7.
        result.append(dict(path=str(path.resolve()), label=parts[2] - 1, actor=parts[6]))
    if not result:
        raise ValueError('No RAVDESS speech WAV files found. Extract the dataset first.')
    return result


def split_records(rows, seed):
    """Assign whole actors to test/validation/train and require every class."""
    # STEP 5: Split PEOPLE, not individual recordings. Otherwise the network can
    # encounter the same voice in training and testing, inflating an unseen-voice
    # claim. Grouping also keeps this dataset's identical actor-07 pair together.
    actors = sorted({r['actor'] for r in rows})
    if len(actors) < 6:
        raise ValueError('At least six actors are required for separate train/validation/test sets.')
    # A separate generator keeps actor assignments independent of model weights
    # and augmentation randomness. A fixed split supports fair model comparison.
    rng = random.Random(seed)
    rng.shuffle(actors)
    # For 24 actors: 4 test + 4 validation + 16 train, not an 80/10/10 split.
    # This one split is cheap but uncertain. Group cross-validation would measure
    # several held-out actor groups; it was not run. Gender is not stratified.
    n = max(1, round(len(actors) / 6))
    groups = dict(test=actors[:n], validation=actors[n:2*n], train=actors[2*n:])
    splits = {key: [r for r in rows if r['actor'] in ids] for key, ids in groups.items()}
    for key, values in splits.items():
        if {r['label'] for r in values} != set(range(8)):
            raise ValueError(f'{key} split does not contain all eight emotions.')
    return splits


class EmotionCNN(nn.Module):
    """Learn local spectrogram patterns, pool them, and return eight logits."""
    # STEP 6: A convolutional neural network learns local time/frequency patterns.
    # CNN = convolutional neural network; a filter is a small trainable array.
    # Unlike MFCC summary statistics, the input retains a time dimension.
    def __init__(self, pool_bands=1):
        """Build four convolution blocks and a frequency-aware classifier."""
        super().__init__()
        layers = []
        channels = 1
        for out in [16, 32, 64, 128]:
            # Conv: learn 3x3 patterns; padding=1 preserves width/height.
            # BatchNorm: normalize activations using batch statistics during
            # training and running estimates at evaluation. ReLU adds nonlinearity.
            # MaxPool: halve both spatial axes. Dropout2d: randomly zero whole
            # feature maps during training to discourage reliance on one detector.
            layers.extend([nn.Conv2d(channels, out, 3, padding=1), nn.BatchNorm2d(out),
                           nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1)])
            channels = out
        # Shape trace: [B,1,64,401] -> [B,16,32,200] -> [B,32,16,100]
        # -> [B,64,8,50] -> [B,128,4,25]. B is the batch size, usually 16.
        # Average time but retain four frequency regions for the selected CNN:
        # [B,128,4,1] -> [B,512]. The baseline averages frequency too -> [B,128].
        # Four regions performed better on validation here; this is not proof
        # that frequency pooling is always harmful on every audio task.
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((pool_bands, 1)), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * pool_bands, 8))

    def forward(self, x):
        """Transform [batch,1,64,401] into [batch,8] with default features."""
        # Return eight raw scores (logits), not softmax probabilities. Training
        # cross entropy already includes log-softmax in a numerically stable form.
        return self.classifier(self.features(x))


class EmotionMLP(nn.Module):
    """Standardize 160 MFCC statistics using saved buffers and classify them."""
    # A multilayer perceptron learns from the 160 MFCC statistics. This gives a
    # handcrafted-feature baseline; WavLM is the separate pretrained branch.
    def __init__(self):
        """Register preprocessing state and build the 256/128-unit MLP."""
        super().__init__()
        # Buffers move with .to(device) and are saved in state_dict, but are not
        # learned by the optimizer. Fit them to TRAINING features in train().
        self.register_buffer('mean', torch.zeros(160))
        self.register_buffer('std', torch.ones(160))
        # LayerNorm normalizes within an example rather than across the batch.
        # Dropout regularizes, but did not prevent this model from overfitting.
        self.network = nn.Sequential(nn.Linear(160, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.35),
                                     nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(.35),
                                     nn.Linear(128, 8))

    def forward(self, x):
        """Broadcast per-feature normalization across a batch and return logits."""
        return self.network((x - self.mean) / self.std)


def make_model(kind='cnn', pool_bands=4, config=None):
    """Construct the trainable network, without restoring its learned weights."""
    # WavLM's encoder lives in FrozenWavLM; this factory builds only its head.
    # Import inside the branch so ordinary CNN/MFCC runs need not import WavLM.
    if kind == 'wavlm':
        from wavlm import WavLMClassifier
        return WavLMClassifier(config['embedding_dim'])
    if kind == 'mfcc':
        return EmotionMLP()
    if kind != 'cnn':
        raise ValueError(f'Unknown model type: {kind}')
    return EmotionCNN(pool_bands)


def load_checkpoint(path, device):
    """Restore (metadata, trained network, matching feature-extraction callable)."""
    # weights_only narrows deserialization; map_location supports CPU/GPU moves.
    # Rebuilding the same architecture must precede loading its state dictionary.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint['config']
    kind = checkpoint.get('model_type', 'cnn')
    model = make_model(kind, checkpoint.get('pool_bands', 1), config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # feature is a function for CNN/MFCC; FrozenWavLM is a callable object that
    # additionally owns a frozen speech encoder. Both accept (path, config).
    extract = feature
    if kind == 'wavlm':
        from wavlm import FrozenWavLM, ENCODER_DIRECTORY
        extract = FrozenWavLM(config, device, Path(path).resolve().parent / ENCODER_DIRECTORY)
    return checkpoint, model, extract


def device_for(name):
    """Resolve CPU/CUDA placement, with an explicit error for unavailable CUDA."""
    # CUDA is PyTorch's NVIDIA GPU backend. The GPU accelerates tensor operations;
    # it does not improve statistical accuracy by itself. librosa preprocessing
    # runs on CPU; WavLM's encoder runs on the selected device, without gradients.
    # Explicit cuda fails if unavailable; auto chooses CUDA or falls back to CPU.
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)


def loader(rows, config, batch_size, shuffle=False, cache=None, extract=None):
    """Cache deterministic features, then return ordered or shuffled minibatches."""
    # STEP 7: Compute expensive audio features once, then cache the arrays.
    # Cache keys include file metadata and settings, NOT the feature source code;
    # rebuild CNN/MFCC caches if feature() changes. WavLM config also includes a
    # manually maintained feature_version, exact encoder revision and pooling.
    if config.get('kind') == 'wavlm' and extract is None:
        raise ValueError('WavLM features require a loaded frozen encoder.')
    extract = extract or feature
    # enumerate(..., 1) provides a human-friendly progress count. The row order
    # remains unchanged; labels are paired with the corresponding feature array.
    values = []
    for index, row in enumerate(rows, 1):
        path = Path(row['path'])
        fingerprint = json.dumps([str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns, config])
        key = hashlib.sha256(fingerprint.encode()).hexdigest()
        # Hash the JSON settings/metadata into a filesystem-safe cache filename.
        # Metadata is not an audio-content hash; the audit does that separately.
        target = Path(cache) / f'{key}.npy' if cache else None
        if target is not None and target.exists():
            value = np.load(target, allow_pickle=False)
        else:
            value = extract(path, config)
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, value)
        if config.get('kind') == 'wavlm':
            if value.shape != (config['embedding_dim'],) or not np.isfinite(value).all():
                raise ValueError(f'Invalid WavLM embedding cache: {target}')
            if index % 100 == 0 or index == len(rows):
                print(f'WavLM features: {index}/{len(rows)}', flush=True)
        values.append(value)
    # Stack creates CNN [N,1,64,401], MFCC [N,160] or WavLM [N,1536] for base-plus.
    # Keep all features in CPU RAM because this dataset is small. A larger
    # dataset would benefit from a lazy Dataset and parallel worker processes.
    x = torch.from_numpy(np.stack(values))
    y = torch.tensor([r['label'] for r in rows], dtype=torch.long)
    # Shuffle only training to vary batch order; evaluation order must match
    # recording manifests for per-file reporting. Zero workers keeps Windows
    # multiprocessing simple. Only each current batch is transferred to the GPU.
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def score(model, batches, device):
    """Return unweighted loss, per-recording scores and eight-class metrics."""
    # STEP 9: Evaluation is forward-only. eval() disables dropout and switches
    # BatchNorm to stored statistics; inference_mode() disables gradient tracking.
    # They solve DIFFERENT problems, so we need both.
    model.eval()
    truth, predictions, probabilities = [], [], []
    loss = 0.0
    with torch.inference_mode():
        for x, y in batches:
            logits = model(x.to(device))
            loss += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
            # argmax chooses the class. Softmax turns logits into nonnegative
            # scores summing to one; the winning class is unchanged. These scores
            # are not calibrated evidence of a person's actual internal emotion.
            predictions.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
            truth.extend(y.tolist())
    # Accuracy = fraction correct. F1 balances precision and recall per class;
    # macro F1 averages the eight class F1 values equally. This protects the
    # smaller neutral class from being overwhelmed by the seven larger classes.
    # zero_division=0 reports 0 when precision/F1 would be undefined.
    # Confusion matrix rows are true labels; columns are predicted labels.
    return dict(loss=loss / len(truth), truth=truth, predictions=predictions, probabilities=probabilities,
                accuracy=accuracy_score(truth, predictions),
                macro_f1=f1_score(truth, predictions, labels=list(range(8)), average='macro', zero_division=0),
                report=classification_report(truth, predictions, labels=list(range(8)), target_names=EMOTIONS,
                                             output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(truth, predictions, labels=list(range(8))).tolist())


def save_json(path, value):
    """Write readable UTF-8 metadata; caller creates the parent directory."""
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def train(args):
    """Learn CNN/MLP weights or only the WavLM head; select using validation."""
    # STEP 8: Fit model weights on training data, use validation for decisions,
    # and reserve test predictions with --skip-test while comparing candidates.
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('Epochs, batch size and patience must be positive.')
    # Seed each randomness source: Python augmentation, NumPy, and PyTorch.
    # This helps repeatability but does not guarantee identical GPU results
    # across hardware/library versions or nondeterministic GPU operations.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    config = dict(CONFIG, kind=args.model)
    output = Path(args.output)
    # Fail on an existing run rather than silently overwrite experiment evidence.
    output.mkdir(parents=True, exist_ok=False)
    splits = split_records(records(args.data), args.split_seed)
    save_json(output / 'splits.json', splits)
    extract = feature
    if args.model == 'wavlm':
        # Download/load one frozen encoder, pin its Hub revision in config, and
        # export an offline copy. It extracts embeddings before head training;
        # AdamW below receives the classifier parameters, not encoder weights.
        from wavlm import FrozenWavLM, ENCODER_DIRECTORY
        config.update(wavlm_checkpoint=args.wavlm_checkpoint, wavlm_revision=args.wavlm_revision)
        print(f'Loading frozen WavLM: {args.wavlm_checkpoint}', flush=True)
        extract = FrozenWavLM(config, device)
        extract.save(output / ENCODER_DIRECTORY)
    save_json(output / 'config.json', dict(arguments=vars(args), preprocessing=config, torch=torch.__version__, device=str(device)))
    print(f'Device: {device}; preparing features', flush=True)
    batches = {k: loader(v, config, args.batch_size, k == 'train', args.cache, extract)
               for k, v in splits.items() if k != 'test'}
    # The comprehension excludes test audio. Dict insertion order may prepare
    # validation features first; no fitting occurs until training features exist.
    model = make_model(args.model, args.pool_bands, config).to(device)
    if args.model in ['mfcc', 'wavlm']:
        # Fit each feature's mean/std on training only. Using validation/test
        # features here would leak information into the fitted preprocessing.
        training_features = batches['train'].dataset.tensors[0]
        # copy_ stores fitted statistics in registered buffers, rather than
        # replacing their tensors. Clamp nearly constant features away from zero.
        model.mean.copy_(training_features.mean(0))
        model.std.copy_(training_features.std(0).clamp_min(1e-5))
    # Neutral has half as many examples, so give its errors twice the weight.
    # Formula: N / (number_of_classes * class_count). We could oversample neutral
    # instead; weighting avoids repeatedly drawing its small set of recordings.
    counts = np.bincount([r['label'] for r in splits['train']], minlength=8)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (8 * counts), dtype=torch.float32, device=device))
    # Cross entropy rewards a high score for the correct class. AdamW adapts
    # parameter update sizes; weight decay discourages overly large weights.
    # SGD with momentum is a valid alternative, but would need its own tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # For WavLM, model means WavLMClassifier. FrozenWavLM is outside this model.
    # Reduce learning rate after validation improvement stalls. This is separate
    # from early stopping: first take smaller steps, later stop trying.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # Mixed precision uses lower precision where appropriate to reduce memory
    # and potentially speed GPU work. Gradient scaling helps avoid tiny float16
    # gradients underflowing. CPU runs use ordinary float32 in this pipeline.
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best, stale = [], -1, 0
    for epoch in range(1, args.epochs + 1):
        # An epoch is one pass through all training examples; a batch is one
        # smaller group producing an optimizer update. CNN defaults: 960/16 = 60;
        # WavLM experiment: 960/32 = 30. The encoder is not rerun each epoch.
        started = time.monotonic()
        model.train()
        total, correct = 0.0, 0
        for x, y in batches['train']:
            x, y = x.to(device), y.to(device)
            # Mask small frequency/time regions only in the training batch.
            if args.model == 'cnn':
                # Simple SpecAugment-style masking: hide 6 mel bands and 20
                # time frames (~0.2 s). Zero means normalized feature mean, not
                # acoustic silence. A single mask location is shared per batch.
                # Noise/pitch/time-stretch augmentation was not tested; strong
                # changes can also alter emotion cues. Validation is unmasked.
                f = random.randrange(x.shape[2] - 6 + 1)
                t = random.randrange(x.shape[3] - 20 + 1)
                x[:, :, f:f+6, :] = 0
                x[:, :, :, t:t+20] = 0
            # Gradients accumulate in PyTorch unless cleared. We want this
            # batch's gradient, not an accidental sum across previous batches.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            # forward: logits -> loss; backward: chain rule computes gradients;
            # step: optimizer changes weights to try to reduce future loss.
            scaler.scale(loss).backward()
            # Unscale BEFORE clipping so the threshold applies to real gradients.
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        # Training accuracy uses dropout and changing weights; only CNN uses masks.
        # Validation uses eval mode and no masks, so higher validation accuracy
        # than training accuracy does not automatically imply a bug or leakage.
        # Training loss is a batch-averaged class-weighted loss; validation loss
        # is unweighted. Compare their trends cautiously, not their raw gap alone.
        metrics = score(model, batches['validation'], device)
        history.append(dict(epoch=epoch, train_loss=total / len(splits['train']),
                            train_accuracy=correct / len(splits['train']), validation_loss=metrics['loss'],
                            validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1'],
                            lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - started))
        scheduler.step(metrics['macro_f1'])
        print(history[-1], flush=True)
        save_json(output / 'history.json', history)
        if metrics['macro_f1'] > best:
            # Save the BEST validation epoch, not necessarily the final one.
            # Keep preprocessing and architecture with weights: inference must
            # reconstruct exactly the same feature layout and model dimensions.
            best, stale = metrics['macro_f1'], 0
            torch.save(dict(model=model.state_dict(), config=config, emotions=EMOTIONS, model_type=args.model,
                            epoch=epoch, seed=args.seed, pool_bands=args.pool_bands,
                            split_seed=args.split_seed), output / 'best.pt')
        else:
            stale += 1
        # Early stopping counts consecutive epochs without a new best score.
        # More epochs can fit training actors better without helping new actors.
        if stale >= args.patience:
            break
    # Restore selected weights; do not accidentally report the last epoch's model.
    # This is an inference checkpoint, not a full exact-resume checkpoint:
    # optimizer/scheduler/scaler/random-generator states are not saved.
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    validation = score(model, batches['validation'], device)
    save_json(output / 'validation_metrics.json', validation)
    metrics = validation
    if not args.skip_test:
        metrics = score(model, loader(splits['test'], config, args.batch_size, cache=args.cache, extract=extract), device)
        save_json(output / 'test_metrics.json', metrics)
        print(f'Test accuracy: {metrics["accuracy"]:.3f}; macro F1: {metrics["macro_f1"]:.3f}')
    plot_results(output, history, metrics, 'Validation' if args.skip_test else 'Test')


def plot_results(output, history, metrics, split='Test'):
    """Delegate numerical history/metrics to the shared offline report renderer."""
    from reporting import training_report
    training_report(Path(output) / 'report.html', history, metrics, split)


def main():
    """Parse one subcommand and dispatch training, evaluation, or prediction."""
    # Command-line entry point: train fits weights, evaluate scores labelled
    # held-out files, predict classifies one file whose true label may be unknown.
    parser = argparse.ArgumentParser(description=__doc__)
    # add_subparsers selects one command. type=int/float converts CLI strings;
    # choices validates supported values; store_true makes a presence-only flag.
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', required=True)
    training.add_argument('--output', default='runs/baseline')
    training.add_argument('--epochs', type=int, default=60)
    training.add_argument('--batch-size', type=int, default=16)
    training.add_argument('--patience', type=int, default=12)
    training.add_argument('--lr', type=float, default=0.001)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--model', choices=['cnn', 'mfcc', 'wavlm'], default='cnn')
    training.add_argument('--wavlm-checkpoint', default='microsoft/wavlm-base-plus', help='Pretrained Hugging Face WavLM encoder ID')
    training.add_argument('--wavlm-revision', default='main', help='Hub revision; resolved commit is saved for reproducibility')
    # Model seed affects initialization/shuffling; split seed affects actor IDs.
    # pool-bands affects CNN only; WavLM-specific flags affect WavLM only.
    training.add_argument('--split-seed', type=int, default=42)
    training.add_argument('--cache', default='data/features')
    training.add_argument('--skip-test', action='store_true', help='Reserve test actors during model development')
    training.add_argument('--pool-bands', type=int, choices=[1, 4], default=4,
                          help='Keep four frequency regions, or use one for a globally pooled baseline')
    for name in ['predict', 'evaluate']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--checkpoint', required=True)
        cmd.add_argument('--audio' if name == 'predict' else '--manifest', required=True)
        cmd.add_argument('--output', help='Save prediction/evaluation JSON and Plotly report')
    for cmd in [training, *[commands.choices[n] for n in ['predict', 'evaluate']]]:
        cmd.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
        return
    device = device_for(args.device)
    checkpoint, model, extract = load_checkpoint(args.checkpoint, device)
    # map_location lets a GPU-trained checkpoint load on CPU. weights_only limits
    # deserialization; still use trusted checkpoints. Defaults support older runs.
    if args.command == 'evaluate':
        rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['test']
        metrics = score(model, loader(rows, checkpoint['config'], 16, cache='data/features', extract=extract), device)
        if args.output:
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            save_json(output / 'test_metrics.json', metrics)
            plot_results(output, [], metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k not in ['truth', 'predictions', 'probabilities']}, indent=2))
    else:
        # Reuse TRAINING preprocessing settings, not newly invented defaults.
        # The extra [None] adds a batch dimension for one recording:
        # CNN [1,64,401] -> [1,1,64,401], MFCC [160] -> [1,160],
        # or base-plus WavLM [1536] -> [1,1536]. to(device) matches model placement.
        x = torch.from_numpy(extract(args.audio, checkpoint['config']))[None].to(device)
        with torch.inference_mode():
            probabilities = model(x).softmax(1)[0].cpu().tolist()
        result = dict(prediction=EMOTIONS[int(np.argmax(probabilities))],
                      probabilities=dict(zip(EMOTIONS, probabilities)))
        if args.output:
            from reporting import prediction_report
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            save_json(output / 'prediction.json', result)
            prediction_report(output / 'prediction.html', args.audio, result, checkpoint.get('model_type', 'cnn'))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    # Importing ravdess exposes helpers without parsing CLI arguments/training.
    main()
