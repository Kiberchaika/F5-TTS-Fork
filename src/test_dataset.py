import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from f5_tts.model.dataset import *
from f5_tts.model.modules import MelSpec
from tqdm import tqdm
import numpy as np
import io
import soundfile as sf
import json
from vocos import Vocos

if __name__ == "__main__":
        
    dataset = RussianSingingDataset(
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    )

    sample = dataset[0]
    print("ok")
