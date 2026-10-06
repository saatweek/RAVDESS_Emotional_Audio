"""Serve three saved emotion models through uploads and microphone streaming.

Study decode_upload -> append_window -> EmotionService.predict -> build_app ->
main. Gradio owns the browser UI and event transport; our callbacks own decoding,
per-session audio buffers, and inference. See CODE_WALKTHROUGH.md for the route.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import threading

# Keep browser uploads inside this project's ignored data folder. Set these
# before importing Gradio so its cache and analytics use the intended defaults.
# setdefault respects an existing environment setting instead of overwriting it.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / 'data' / 'gradio_cache'))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch

from ravdess import EMOTIONS, device_for, load_checkpoint

# Match the rate saved by training. Four seconds is 64,000 waveform samples.
SAMPLE_RATE = 16000
WINDOW_SECONDS = 4
MAX_UPLOAD_SECONDS = 30
# Paths are relative to this script, so serving does not depend on the shell's
# working directory. Each entry is a trained emotion checkpoint, not a raw model.
MODEL_PATHS = {
    'CNN (spectrogram)': ROOT / 'runs/final/best.pt',
    'MFCC + neural network': ROOT / 'runs/mfcc_seed42/best.pt',
    'WavLM': ROOT / 'runs/wavlm_comparison/best.pt',
}
AUDIO_SUFFIXES = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac', '.aac', '.aiff', '.aif', '.mp4'}


def decode_upload(path):
    """Decode an accepted file to finite mono float32 samples at 16 kHz.

    Browser microphone chunks also enter here as file paths. The bundled FFmpeg
    handles compressed containers without a separate system installation.
    """
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError('Upload a WAV, MP3, M4A, WebM, OGG, FLAC, AAC or AIFF audio file.')
    if path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError('Please upload an audio file smaller than 25 MB.')
    # Pass arguments as a list, without a shell. Restrict protocols/formats, cap
    # decoding at 31 seconds, and return raw little-endian float32 via stdout.
    # The extra second lets us reject a >30-second upload rather than silently
    # presenting a truncated recording as if the entire upload was analyzed.
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
         '-protocol_whitelist', 'file,pipe',
         '-format_whitelist', 'wav,mp3,mov,matroska,webm,ogg,flac,aac,aiff',
         '-i', str(path), '-t', str(MAX_UPLOAD_SECONDS + 1),
         '-f', 'f32le', '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1'],
        capture_output=True, timeout=20, check=False)
    if result.returncode:
        raise ValueError('Could not decode this file. Try a WAV or MP3 recording.')
    # copy() creates writable array storage independent of the captured bytes.
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(audio) > MAX_UPLOAD_SECONDS * SAMPLE_RATE:
        raise ValueError(f'Please upload a clip no longer than {MAX_UPLOAD_SECONDS} seconds.')
    return validate_samples(audio)


def validate_samples(audio):
    """Require a nonempty, finite, one-dimensional waveform; keep float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('The recording is empty or contains invalid audio.')
    return audio


def microphone_samples(chunk):
    """Normalize a (rate, samples) pair, downmix channels, and resample.

    Our live callback supplies the pair after decoding Gradio's filepath input.
    This helper also accepts PCM arrays, which the tests exercise directly.
    """
    rate, samples = chunk
    if not isinstance(rate, (int, np.integer)) or not 8000 <= rate <= 192000:
        raise ValueError('Unsupported microphone sample rate.')
    samples = np.asarray(samples)
    if samples.ndim not in (1, 2) or samples.shape[0] > rate * 10:
        raise ValueError('Invalid microphone audio chunk.')
    # PCM integers encode amplitude on an integer scale; float audio is already
    # amplitude-valued. Unsigned 8-bit PCM has its zero point at 128.
    if np.issubdtype(samples.dtype, np.signedinteger):
        samples = samples.astype(np.float32) / float(-np.iinfo(samples.dtype).min)
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128) / 128
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        if not 1 <= samples.shape[1] <= 8:
            raise ValueError('Unsupported number of microphone channels.')
        # Here stereo is [samples, channels], unlike librosa's channel-first load.
        samples = samples.mean(axis=1)
    samples = validate_samples(samples)
    return librosa.resample(samples, orig_sr=int(rate), target_sr=SAMPLE_RATE)


def append_window(chunk, previous):
    """Append received samples and retain at most the latest 64,000 of them."""
    samples = microphone_samples(chunk)
    previous = np.empty(0, dtype=np.float32) if previous is None else previous
    return np.concatenate([previous, samples])[-WINDOW_SECONDS * SAMPLE_RATE:]


class EmotionService:
    """Load models once, share their weights, and serialize inference calls.

    Audio history belongs to gr.State in each browser session, not this object.
    The lock covers preprocessing as well as model execution, including WavLM.
    """
    def __init__(self, device='auto', paths=None):
        """Restore architecture, normalization buffers, and encoder if needed."""
        self.device = device_for(device)
        self.models = {}
        self.lock = threading.Lock()
        for name, path in (MODEL_PATHS if paths is None else paths).items():
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing {name} checkpoint: {path}')
            print(f'Loading {name} on {self.device}', flush=True)
            checkpoint, model, extract = load_checkpoint(path, self.device)
            # A label-order mismatch would attach the wrong names to the scores.
            if checkpoint['emotions'] != EMOTIONS:
                raise ValueError(f'{name} uses a different emotion label order.')
            self.models[name] = (checkpoint, model, extract)

    def predict(self, audio, name):
        """Apply the checkpoint's preprocessing and return eight softmax scores."""
        if name not in self.models:
            raise ValueError('Choose one of the available models.')
        audio = validate_samples(audio)
        if len(audio) < SAMPLE_RATE // 4:
            raise ValueError('Please provide at least a quarter-second of audio.')
        if np.max(np.abs(audio)) < 1e-8:
            raise ValueError('No sound detected. Please speak into the microphone.')
        checkpoint, model, extract = self.models[name]
        # Reuse the exact saved preprocessing through a float WAV. Temporary
        # input files are removed even if preprocessing or prediction fails.
        with self.lock, tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'clip.wav'
            sf.write(path, audio, SAMPLE_RATE, subtype='FLOAT')
            # [None] adds batch dimension: [1,1,64,401], [1,160], or [1,1536].
            # The model returns logits; softmax converts each row to sum to one.
            x = torch.from_numpy(extract(path, checkpoint['config']))[None].to(self.device)
            with torch.inference_mode():
                scores = model(x).softmax(1)[0].cpu().tolist()
        return dict(zip(EMOTIONS, scores))


def build_app(service):
    """Create components, connect callbacks, and configure Gradio's queue."""
    def upload_predict(path, name):
        """Convert expected input failures to a friendly browser error."""
        if path is None:
            raise gr.Error('Choose an audio file first.')
        try:
            scores = service.predict(decode_upload(path), name)
        except (ValueError, subprocess.TimeoutExpired) as error:
            raise gr.Error(str(error)) from error
        return scores, 'Prediction complete.'

    def stream_predict(chunk, name, buffer):
        """Return (scores, updated session buffer, status) for a received chunk."""
        if chunk is None:
            return gr.skip(), buffer, 'Waiting for microphone audio...'
        try:
            # Keep Gradio's file input unconverted; the bundled decoder also
            # handles compressed microphone containers on mobile browsers.
            chunk = (SAMPLE_RATE, decode_upload(chunk))
            buffer = append_window(chunk, buffer)
            if len(buffer) < WINDOW_SECONDS * SAMPLE_RATE:
                return {}, buffer, f'Listening: {len(buffer) / SAMPLE_RATE:.1f} / {WINDOW_SECONDS} seconds...'
            # This fixed RMS threshold suppresses very quiet windows. It is not
            # learned voice detection and does not distinguish speech from noise.
            if np.sqrt(np.mean(buffer ** 2)) < 1e-4:
                return {}, buffer, 'Audio is very quiet. Keep speaking or move closer to the microphone.'
            return service.predict(buffer, name), buffer, 'Live prediction from your latest four seconds of audio.'
        except (ValueError, subprocess.TimeoutExpired) as error:
            return {}, None, str(error)

    # Component construction declares the page; callbacks run later on events.
    # Cache cleanup runs hourly and targets files older than an hour, not instantly.
    with gr.Blocks(title='Speech Emotion Demo', delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# Speech Emotion Demo\nChoose a model, then upload a clip or speak into your microphone.')
        model = gr.Dropdown(list(service.models), value='WavLM' if 'WavLM' in service.models else next(iter(service.models)),
                            label='Model', interactive=True)
        with gr.Tabs():
            with gr.Tab('Upload audio'):
                upload = gr.File(label='Audio clip (up to 30 seconds)', file_types=['audio'], type='filepath')
                predict = gr.Button('Predict emotion', variant='primary')
            with gr.Tab('Microphone — live'):
                gr.Markdown('Press the microphone record button and keep speaking. The first result appears after '
                            'about four seconds, then updates while you talk. Press stop when finished.')
                mic = gr.Audio(sources=['microphone'], type='filepath', format=None,
                               streaming=True, label='Microphone')
                # Per-session state prevents one visitor's audio joining another's.
                buffer = gr.State(value=None, time_to_live=3600)
        result = gr.Label(label='Predicted speech emotion', num_top_classes=8)
        status = gr.Textbox(label='Status', value='Ready.', interactive=False)
        gr.Markdown('Models were trained on acted RAVDESS speech. Scores describe model predictions and '
                    'are not calibrated confidence or a measurement of your mental state.')
        predict.click(upload_predict, [upload, model], [result, status], api_name='predict_clip',
                      concurrency_id='inference', concurrency_limit=1)
        # Streaming events need their own scheduler group: a long-running stream
        # occupies a scheduler slot. The shared lock still serializes inference;
        # separate groups do not guarantee fairness or bounded end-to-end latency.
        # stream_every requests ~1-second chunks; time_limit is a queue scheduling
        # budget, not an automatic stop after 30 seconds of microphone recording.
        live_event = mic.stream(stream_predict, [mic, model, buffer], [result, buffer, status],
                                stream_every=1, time_limit=30, concurrency_limit=4, api_visibility='private')
        # Reset state on a new recording. Stop preserves the last displayed result.
        # Cancellation cannot undo an inference callback already executing.
        mic.start_recording(lambda: (None, {}, 'Listening...'), outputs=[buffer, result, status],
                            queue=False, api_visibility='private')
        mic.clear(lambda: (None, {}, 'Ready.'), outputs=[buffer, result, status], queue=False,
                  cancels=[live_event], api_visibility='private')
        mic.stop_recording(lambda: (None, 'Recording stopped. Press record to start again.'), outputs=[buffer, status],
                           queue=False, cancels=[live_event], api_visibility='private')
        # Switching models clears the display but keeps the session audio buffer.
        model.change(lambda: ({}, 'Ready. Upload a clip or keep speaking.'), outputs=[result, status],
                     queue=False, api_visibility='private')
    return demo.queue(max_size=16, default_concurrency_limit=1)


def main():
    """Choose device/port, restore and warm models, then run the web server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--share', action='store_true', help='Create a public HTTPS Gradio link')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    service = EmotionService(args.device)
    # Exercise preprocessing and CUDA kernels before visitors make requests.
    # This generated tone checks readiness; it is not an emotion example.
    print('Warming up the three models...', flush=True)
    t = np.arange(WINDOW_SECONDS * SAMPLE_RATE) / SAMPLE_RATE
    for name in service.models:
        service.predict((.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), name)
    app = build_app(service)
    # Loopback is local to the laptop; --share adds a public HTTPS relay URL.
    # Restrict Gradio's file serving for weights/repository/environment folders.
    app.launch(server_name='127.0.0.1', server_port=args.port, share=args.share,
               max_file_size='25mb', show_error=False,
               blocked_paths=[str(ROOT / 'runs'), str(ROOT / '.git'), str(ROOT / '.venv')])


if __name__ == '__main__':
    main()
