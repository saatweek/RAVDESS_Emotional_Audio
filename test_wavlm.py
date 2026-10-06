"""Three local WavLM integration checks; no Hub downloads or accuracy claim.

A randomly initialized tiny encoder exercises the same Hugging Face API as the
pretrained model without its memory/download cost. See CODE_WALKTHROUGH.md.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor

from ravdess import CONFIG, EMOTIONS, loader, load_checkpoint, make_model
from wavlm import FrozenWavLM, waveform


class WavLMTests(unittest.TestCase):
    """Check freeze semantics, saved encoder/head integration, and input/cache rules."""
    def test_frozen_encoder_and_offline_prediction_roundtrip(self):
        """Update the head while freezing the encoder, then restore both offline."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            encoder_dir = root / 'wavlm_encoder'
            # Eight hidden channels produce 16 mean/std features, not base-plus's
            # 1,536. This is an API/shape test, not pretrained speech recognition.
            tiny = WavLMConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=16, conv_dim=(8, 8, 8),
                               conv_kernel=(10, 3, 3), conv_stride=(5, 2, 2),
                               num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
            WavLMModel(tiny).save_pretrained(encoder_dir)
            Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True,
                                    do_normalize=False).save_pretrained(encoder_dir)
            config = dict(CONFIG, kind='wavlm', embedding_dim=16)
            extractor = FrozenWavLM(config, 'cpu', encoder_dir)
            self.assertFalse(extractor.encoder.training)
            self.assertTrue(all(not p.requires_grad for p in extractor.encoder.parameters()))
            audio = root / 'audio.wav'
            t = np.arange(4410) / 22050
            sf.write(audio, np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 22050)
            # Resampling stereo works, short clips retain actual length, and
            # repeated extraction disables encoder dropout/SpecAugment.
            self.assertEqual(waveform(audio, config).shape, (3200,))
            embedding = extractor(audio, config)
            np.testing.assert_array_equal(embedding, extractor(audio, config))
            self.assertEqual(embedding.shape, (16,))
            model = make_model('wavlm', config=config).eval()
            model.mean.fill_(.1)
            model.std.fill_(2)
            x = torch.from_numpy(embedding)[None]
            # eval() disables dropout, not gradients; the head can still receive
            # this optimizer update while the speech encoder remains frozen.
            optimizer = torch.optim.AdamW(model.parameters())
            loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([3]))
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None for p in extractor.encoder.parameters()))
            checkpoint_path = root / 'best.pt'
            torch.save(dict(model=model.state_dict(), config=config, model_type='wavlm', emotions=EMOTIONS), checkpoint_path)
            _, restored, restored_extractor = load_checkpoint(checkpoint_path, 'cpu')
            with torch.inference_mode():
                expected = model(x)
                actual = restored(torch.from_numpy(restored_extractor(audio, config))[None])
            torch.testing.assert_close(expected, actual)
            self.assertEqual(tuple(actual.shape), (1, 8))

    def test_cache_reuses_features_and_invalidates_on_revision(self):
        """Mock extraction counts distinguish cache reuse from revision misses."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / 'clip.wav'
            audio.touch()
            rows = [dict(path=str(audio), label=2)]
            config = dict(CONFIG, kind='wavlm', embedding_dim=16, wavlm_revision='commit-a')
            extract = Mock(return_value=np.ones(16, dtype=np.float32))
            loader(rows, config, 1, cache=root / 'cache', extract=extract)
            loader(rows, config, 1, cache=root / 'cache', extract=extract)
            self.assertEqual(extract.call_count, 1)
            loader(rows, dict(config, wavlm_revision='commit-b'), 1, cache=root / 'cache', extract=extract)
            self.assertEqual(extract.call_count, 2)
            with self.assertRaisesRegex(ValueError, 'loaded frozen encoder'):
                loader(rows, config, 1, cache=root / 'cache')

    def test_crop_and_invalid_audio(self):
        """Verify center slicing and reject silence, undersized, and nonfinite audio."""
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / 'clip.wav'
            config = dict(CONFIG, kind='wavlm')
            samples = np.sin(np.arange(80000) * .1).astype(np.float32)
            sf.write(audio, samples, 16000, subtype='FLOAT')
            np.testing.assert_allclose(waveform(audio, config), samples[8000:72000])
            for samples, message in [(np.zeros(1600), 'silent'), (np.ones(100), 'too short'),
                                     (np.full(1600, np.nan), 'invalid')]:
                sf.write(audio, samples, 16000, subtype='FLOAT')
                with self.assertRaisesRegex(ValueError, message):
                    waveform(audio, config)


if __name__ == '__main__':
    unittest.main()
