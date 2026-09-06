"""Pinned CAM++ Chinese/English ONNX inference; CPU only, no recording or writes.

Model: iic/speech_campplus_sv_zh_en_16k-common_advanced, exported by sherpa-onnx.
Features follow 3D-Speaker's FBank(mean_nor=True): normalized 16k waveform,
80-bin Kaldi filterbanks, global mean subtraction. No legacy score offset.
"""
from pathlib import Path
import hashlib
import os
import urllib.request
import numpy as np

MODEL_URL = ('https://github.com/k2-fsa/sherpa-onnx/releases/download/'
             'speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx')
MODEL_SHA256 = 'aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2'
DIM = 192


def download(path):
    """Atomic, checksum-verified download; never overwrites profiles."""
    path = Path(path)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.download')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as src, tmp.open('wb') as dst:
            while chunk := src.read(1024*1024):
                dst.write(chunk)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('CAM++ model checksum mismatch')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


class Encoder:
    def __init__(self, path, threads=2):
        import onnxruntime as ort
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'CAM++ model missing: {path}; run tools/download_campplus.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('CAM++ model is not the pinned checkpoint; download it again')
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        meta = self.session.get_modelmeta().custom_metadata_map
        if (meta.get('output_dim') != str(DIM) or meta.get('normalize_samples') != '1'
                or meta.get('sample_rate') != '16000'
                or meta.get('feature_normalize_type') != 'global-mean'):
            raise ValueError('Unexpected CAM++ feature/embedding contract')

    def embed(self, audio, sample_rate):
        import torch
        from torchaudio.compliance.kaldi import fbank
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if wav.ndim != 1 or not np.isfinite(wav).all():
            return None
        if sample_rate <= 0 or not len(wav):
            return None
        if sample_rate != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            divisor = gcd(int(sample_rate), 16000)
            wav = resample_poly(wav, 16000//divisor, int(sample_rate)//divisor).astype(np.float32)
        # Remove capture padding, preserving interior pauses and a 60ms margin.
        frame = 160
        n = len(wav)//frame
        if n < 20:
            return None
        rms = np.sqrt(np.mean(wav[:n*frame].reshape(n,frame)**2, axis=1))
        active = np.flatnonzero(rms > max(.001, float(rms.max())*.05))
        if not len(active):
            return None
        wav = wav[max(0,int(active[0]*frame)-960):min(len(wav),int((active[-1]+1)*frame)+960)]
        if len(wav) < 3200:
            return None
        with torch.inference_mode():
            features = fbank(torch.from_numpy(wav.copy()).unsqueeze(0),
                             num_mel_bins=80, sample_frequency=16000, dither=0.0)
            features = features - features.mean(dim=0, keepdim=True)
            arr = features.unsqueeze(0).numpy()
        embedding = self.session.run(['embedding'], {'x': arr})[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if embedding.shape != (DIM,) or not np.isfinite(embedding).all() or norm <= 1e-10:
            return None
        return embedding/norm
