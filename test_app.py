"""Four checks of Gradio callbacks, format decoding, and session audio buffers.

These call Python callbacks directly and generate files; they do not request a
real browser microphone or prove mobile permission/network behavior.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from app import SAMPLE_RATE, append_window, build_app, decode_upload, microphone_samples


class WebAudioTests(unittest.TestCase):
    """Mock inference where possible; use the real bundled FFmpeg for decoding."""
    def test_live_callback_waits_for_context_and_resets_between_recordings(self):
        """Accumulate four seconds, infer on bounded windows, then reset context."""
        # The fake score isolates callback/state behavior from trained accuracy.
        service = Mock(models={'WavLM': None})
        service.predict.return_value = {'neutral': 1.0}
        demo = build_app(service)
        stream = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == 'stream_predict')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'chunk.wav'
            sf.write(path, .1 * np.sin(np.arange(SAMPLE_RATE) * .1), SAMPLE_RATE)
            state = None
            for _ in range(3):
                scores, state, _ = stream(str(path), 'WavLM', state)
                self.assertEqual(scores, {})
            service.predict.assert_not_called()
            scores, state, _ = stream(str(path), 'WavLM', state)
            self.assertEqual(scores, {'neutral': 1.0})
            self.assertEqual(len(state), 4 * SAMPLE_RATE)
            self.assertEqual(service.predict.call_count, 1)
            _, state, _ = stream(str(path), 'WavLM', state)
            self.assertEqual(len(state), 4 * SAMPLE_RATE)
            self.assertEqual(service.predict.call_count, 2)
            _, new_state, message = stream(str(path), 'WavLM', None)
            self.assertEqual(len(new_state), SAMPLE_RATE)
            self.assertIn('Listening', message)
            self.assertEqual(service.predict.call_count, 2)

    def test_resampling_integer_stereo_and_bounded_session_buffers(self):
        """PCM scale/channel/rate conversion and separate state preserve audio meaning."""
        chunk = (48000, np.full((48000, 2), 16384, dtype=np.int16))
        mono = microphone_samples(chunk)
        self.assertEqual(len(mono), SAMPLE_RATE)
        np.testing.assert_allclose(mono[50:-50], .5, atol=.01)
        first = None
        for _ in range(7):
            first = append_window(chunk, first)
        self.assertEqual(len(first), 4 * SAMPLE_RATE)
        # A second visitor's recording must not inherit the first visitor's audio.
        second = append_window((SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.float32)), None)
        self.assertEqual(len(second), SAMPLE_RATE)
        self.assertTrue(np.all(second == 0))
        self.assertGreater(first.mean(), .4)

    def test_decode_wav_and_browser_webm(self):
        """Decode both WAV and an Opus WebM as used by some browser recorders."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav = root / 'clip.wav'
            t = np.arange(24000) / 24000
            sf.write(wav, .2 * np.sin(2 * np.pi * 220 * t), 24000)
            decoded = decode_upload(wav)
            self.assertEqual(len(decoded), SAMPLE_RATE)
            self.assertTrue(np.isfinite(decoded).all())
            webm = root / 'clip.webm'
            # Compressed Opus has codec framing; permit a small length difference.
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                            '-i', str(wav), '-c:a', 'libopus', str(webm)], check=True, timeout=20)
            decoded_webm = decode_upload(webm)
            self.assertLess(abs(len(decoded_webm) - SAMPLE_RATE), 320)
            self.assertGreater(np.max(np.abs(decoded_webm)), .1)

    def test_empty_invalid_and_long_uploads(self):
        """Reject malformed samples/rates/channels and clips exceeding the duration cap."""
        for chunk in [(16000, np.array([])), (16000, np.array([np.nan])),
                      (0, np.ones(100)), (16000, np.ones((10, 20)))]:
            with self.assertRaises(ValueError):
                microphone_samples(chunk)
        with tempfile.TemporaryDirectory() as temp:
            wav = Path(temp) / 'long.wav'
            sf.write(wav, np.ones(SAMPLE_RATE * 31), SAMPLE_RATE)
            with self.assertRaisesRegex(ValueError, '30 seconds'):
                decode_upload(wav)


if __name__ == '__main__':
    unittest.main()
