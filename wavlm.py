"""Frozen Hugging Face WavLM features and a trainable emotion classifier.

The speech encoder stays in evaluation mode. Only the small classifier is
trained; cached embeddings are consequently valid across classifier epochs.
Read waveform -> FrozenWavLM.__init__ -> __call__ -> WavLMClassifier, then
return to ravdess.train. See CODE_WALKTHROUGH.md for the statement walkthrough.
"""
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

# This sibling directory travels with best.pt; the head alone cannot encode WAVs.
ENCODER_DIRECTORY = 'wavlm_encoder'
# Descriptive default constant; the CLI currently repeats this ID explicitly.
DEFAULT_CHECKPOINT = 'microsoft/wavlm-base-plus'
# Increment when changing embedding semantics so newly trained runs get new keys.
FEATURE_VERSION = 1


def waveform(path, config):
    """16 kHz mono, trim edges, center-crop; never pad pooled speech features."""
    if config['sr'] != 16000 or config['seconds'] <= 0:
        raise ValueError('WavLM requires 16 kHz audio and a positive crop duration.')
    # Preserve source rate/channels initially so invalid values can be rejected
    # before mono conversion/resampling. librosa stereo shape is [channels,time].
    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Empty or invalid audio: {path}')
    if np.max(np.abs(audio)) < 1e-8:
        raise ValueError(f'Audio is silent: {path}')
    audio = librosa.to_mono(audio)
    # Unlike changing a header, resampling computes samples at the target rate.
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config['sr'])
    audio, _ = librosa.effects.trim(audio, top_db=35)
    # Discard trim's edge indices; retain the waveform. Center only long clips:
    # max(0, ...) keeps the start at zero for clips shorter than the duration cap.
    size = int(config['sr'] * config['seconds'])
    start = max(0, (len(audio) - size) // 2)
    audio = audio[start:start + size].astype(np.float32)
    # The base encoder's convolutional receptive field is 400 samples.
    if len(audio) < 400:
        raise ValueError(f'Audio is too short for WavLM (minimum 25 ms): {path}')
    return audio


class FrozenWavLM:
    """One unpadded clip per encoder call keeps memory small on a 4 GB GPU."""
    def __init__(self, config, device, directory=None):
        """Load a pinned Hub encoder for training or a saved local one for inference."""
        # Lazy imports keep Hugging Face-specific dependencies in this branch.
        # AutoConfig describes architecture; AutoFeatureExtractor prepares audio;
        # WavLMModel supplies the pretrained encoder, without an emotion head.
        try:
            import transformers
            from transformers import AutoConfig, AutoFeatureExtractor, WavLMModel
        except ImportError as error:
            raise RuntimeError('Install WavLM dependencies with pip install -r requirements.txt.') from error
        self.config = config
        # config is shared with the training caller; resolved metadata is written
        # back into that dictionary before feature cache keys/checkpoints are made.
        self.device = torch.device(device)
        if directory is not None:
            # Inference uses the exported local files. local_files_only prevents
            # silent network downloads or an accidental move to a newer revision.
            source = Path(directory)
            if not source.is_dir():
                raise FileNotFoundError(f'Missing WavLM encoder: {source}. Keep wavlm_encoder/ beside best.pt.')
            source = str(source)
            options = dict(local_files_only=True)
            encoder_config = AutoConfig.from_pretrained(source, **options)
        else:
            source = config['wavlm_checkpoint']
            options = dict(cache_dir='data/huggingface', revision=config.get('wavlm_revision', 'main'))
            encoder_config = AutoConfig.from_pretrained(source, **options)
            # Resolve a moving Hub reference once, then load everything from the
            # same commit. The immutable revision also participates in cache keys.
            revision = encoder_config._commit_hash
            if not revision:
                raise ValueError('Use a Hugging Face WavLM checkpoint ID with a resolvable revision.')
            options['revision'] = revision
            config.update(wavlm_revision=revision, feature_version=FEATURE_VERSION,
                          pooling='last_hidden_mean_std', encoder_precision='float32',
                          transformers_version=transformers.__version__)
        if encoder_config.model_type != 'wavlm':
            # A repository name alone cannot prove the architecture is WavLM.
            raise ValueError('The selected checkpoint must use the WavLM architecture.')
        self.processor = AutoFeatureExtractor.from_pretrained(source, **options)
        if self.processor.sampling_rate != config['sr']:
            raise ValueError('Encoder and audio sample rates must match.')
        self.encoder = WavLMModel.from_pretrained(source, config=encoder_config, **options)
        self.encoder.to(self.device).eval().requires_grad_(False)
        # These solve three different problems: device placement, disabled
        # training-time stochastic layers, and frozen parameter gradients.
        dimension = 2 * self.encoder.config.hidden_size
        if directory is not None and config['embedding_dim'] != dimension:
            raise ValueError('Saved encoder and classifier feature dimensions differ.')
        config['embedding_dim'] = dimension

    def save(self, directory):
        """Export encoder weights/config and processor settings for offline use."""
        # safetensors stores encoder tensors; save_pretrained also writes config.
        self.encoder.save_pretrained(directory, safe_serialization=True)
        self.processor.save_pretrained(directory)

    def __call__(self, path, config):
        """Encode one raw clip into a float32 mean/std vector (base-plus: 1536)."""
        if config != self.config:
            raise ValueError('WavLM feature settings must match the loaded encoder.')
        audio = waveform(path, config)
        inputs = self.processor(audio, sampling_rate=config['sr'], return_tensors='pt', padding=False)
        # The processor's saved configuration controls normalization and mask
        # creation. No tokenizer/text is needed for emotion classification.
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.encoder(**inputs).last_hidden_state[0].float()
            # **inputs unpacks input_values/attention_mask keyword tensors.
            # Output [1,T,768] -> [T,768]; T is the downsampled frame count.
            # Mean captures the clip-level representation; std preserves temporal
            # variability. No padding frames enter either statistic.
            pooled = torch.cat([hidden.mean(0), hidden.std(0, unbiased=False)])
        value = pooled.cpu().numpy().copy()
        # Move off GPU for NumPy; copy gives the caller independent CPU storage.
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite WavLM embedding: {path}')
        return value


class WavLMClassifier(nn.Module):
    """Normalization is fitted on training actors and saved with the weights."""
    def __init__(self, embedding_dim):
        """Create train-fitted buffers and the small learnable emotion network."""
        super().__init__()
        # Register buffers so .to(device) and state_dict include train-fitted
        # statistics. These are not parameters and AdamW does not learn them.
        self.register_buffer('mean', torch.zeros(embedding_dim))
        self.register_buffer('std', torch.ones(embedding_dim))
        self.network = nn.Sequential(nn.Linear(embedding_dim, 256), nn.LayerNorm(256),
                                     nn.ReLU(), nn.Dropout(.35), nn.Linear(256, 8))
        # LayerNorm operates per example; dropout operates during head training.
        # Output is eight logits. CrossEntropyLoss handles log-softmax itself.

    def forward(self, x):
        """Map a standardized embedding batch to eight unnormalized class scores."""
        # Broadcasting applies the saved [embedding_dim] statistics to every row.
        return self.network((x - self.mean) / self.std)
