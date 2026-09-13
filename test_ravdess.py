"""Run with python -m unittest -v; CUDA test skips on CPU-only machines."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ravdess import CONFIG, EmotionCNN, EmotionMLP, feature, records, split_records


class PipelineTests(unittest.TestCase):
    # These are implementation checks, not evidence of real-world accuracy.
    # Held-out evaluation in RESULTS.md answers a different question: how well
    # the trained model predicts labels for actors excluded from training.
    def test_actor_splits_are_disjoint_and_repeatable(self):
        # Synthetic metadata is enough to test grouping: no audio/GPU is needed.
        rows = [dict(actor=a, label=e, path=f'{a}-{e}') for a in range(1, 25) for e in range(8)]
        splits = split_records(rows, 42)
        self.assertEqual(splits, split_records(rows, 42))
        groups = [{r['actor'] for r in splits[k]} for k in ['train', 'validation', 'test']]
        self.assertEqual([len(g) for g in groups], [16, 4, 4])
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_parser_filters_song_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = '03-01-06-01-02-01-12.wav'
            (root / name).touch()
            (root / '03-02-06-01-02-01-12.wav').touch()
            rows = records(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['label'], rows[0]['actor']), (5, 12))
            (root / 'copy').mkdir()
            (root / 'copy' / name).touch()
            with self.assertRaises(ValueError):
                records(root)

    def test_audio_shapes_and_checkpoint_roundtrip(self):
        # Generated tones test resampling/stereo handling and fixed shapes. They
        # do not represent emotions. Saving/reloading should preserve predictions.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'clip.wav'
            features = []
            for seconds in [0.2, 5]:
                t = np.arange(int(22050 * seconds)) / 22050
                sf.write(path, np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 22050)
                features.append(feature(path, CONFIG))
            self.assertEqual(features[0].shape, features[1].shape)
            self.assertTrue(np.isfinite(features).all())
            model = EmotionCNN().eval()
            x = torch.from_numpy(np.stack(features))
            with torch.no_grad():
                expected = model(x)
            checkpoint = Path(temp) / 'model.pt'
            torch.save(model.state_dict(), checkpoint)
            restored = EmotionCNN().eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(x))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mixed_precision_training_step(self):
        # Actually compute a loss, gradients and a GPU update. Merely checking
        # cuda.is_available() would not verify the mixed precision training path.
        model = EmotionCNN().cuda()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler('cuda')
        with torch.autocast('cuda'):
            loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 1, 64, 401, device='cuda')),
                                                     torch.tensor([0, 7], device='cuda'))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(torch.isfinite(loss).item())

    def test_mfcc_and_saved_normalization(self):
        # Weights alone are insufficient if input scaling is lost on reload.
        # This checks that the MFCC model's preprocessing buffers survive too.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            sf.write(path, np.sin(np.arange(16000) * .1), 16000)
            x = torch.from_numpy(feature(path, dict(CONFIG, kind='mfcc')))[None]
            self.assertEqual(tuple(x.shape), (1, 160))
            model = EmotionMLP().eval()
            model.mean.fill_(2)
            model.std.fill_(3)
            checkpoint = Path(temp) / 'weights.pt'
            torch.save(model.state_dict(), checkpoint)
            restored = EmotionMLP().eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x))

    def test_silent_audio_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'silent.wav'
            sf.write(path, np.zeros(1600), 16000)
            with self.assertRaisesRegex(ValueError, 'silent'):
                feature(path, CONFIG)


if __name__ == '__main__':
    unittest.main()
