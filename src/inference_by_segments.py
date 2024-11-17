from __future__ import annotations

import sys
import torch
import os
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import librosa
import random
import cyrtranslit

from f5_tts.infer.utils_infer import (
    infer_single_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text_segment
)

from f5_tts.model import DiT, UNetT

from utils import split_text

import whisperx
import librosa
import numpy as np  

from sentence_ranking import find_best_match   

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisperx.load_model("large-v2", device, compute_type="float16", asr_options = { "hotwords":None})

def normalize_audio(audio):
   """
   Normalize audio to [-1, 1] range
   """
   if len(audio.shape) > 1:
       audio = audio.flatten()
   max_val = np.abs(audio).max()
   if max_val > 0:
       audio = audio / max_val
   return audio

# Load refs
from russian_songs_dataset_utils import process_music_ref_dataset
base_ref_tracks = process_music_ref_dataset("/media/k4_nas/Datasets/Music_RU/Vocal_Dereverb", cache_filename="ref_data_cache.json")#, rewrite_cache=True)
print(f"found {len(base_ref_tracks)} base tracks with lyrics")

# Direct variable assignments instead of command line arguments
model = "F5-TTS"
ckpt_file = "/home/k4/Python/F5-TTS-Fork/ckpts/russian_coarse_16nov/model_0230.pt"

# Default configurations
config_path = os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml")
config = tomli.load(open(config_path, "rb"))

# Additional settings with default values
vocab_file = ""
output_dir = config["output_dir"]
remove_silence = config.get("remove_silence", False)
speed = 1.0
vocoder_name = "vocos"
load_vocoder_from_local = False
wave_path = Path(output_dir) / "infer_cli_out.wav"

# Vocoder settings
if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
mel_spec_type = vocoder_name

vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=load_vocoder_from_local, local_path=vocoder_local_path)

# Model configuration
if model == "F5-TTS":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)

ckpt_file = "/home/k4/Python/F5-TTS-Fork/ckpts/russian_dataset_ft_translit_pinyin/model_last.pt"
ema_model_fine = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)


def inference_by_segments(ref_track_name, gen_text, final_path):  

    ref_track = base_ref_tracks[ref_track_name]

    section = random.choice(ref_track['sections'])

    mp3_path = section['mp3_path']
    words = section['words']
    end_time = section['end_time']

    # Parameters for experiment
    split_step = 4 # Split at step  
    steps = 32 # Steps for generation
    speed = 1.0

    # Open and cut the audio from ref
    audio, sr = librosa.load(mp3_path, sr=None)  
    audio_cut = audio[:int(end_time * sr)]
    sf.write("ref.mp3", audio_cut, sr, format='mp3')

    # Set ref text and cutted audio
    ref_text = cyrtranslit.to_latin(words + " ", "ru").lower()
    ref_audio = "ref.mp3"

    # Preprocess reference audio/text once
    ref_audio_preprocessed, ref_text_preprocessed = ref_audio, ref_text

    splitted_text = split_text(gen_text)
    audio_segments  = []
    for i, text in enumerate(splitted_text):
        print(f"Split {i+1}:")
        print(text)

        ref_audio_preprocessed, ref_text_preprocessed = preprocess_ref_audio_text_segment( ref_audio, ref_text)
        text = cyrtranslit.to_latin(gen_text, "ru").lower() + " "

        max_score = -1

        for j in range(50):

            # Randomize speed
            speed = 0.3 + 0.7 * torch.rand(1)

            # First stage - generate first 16 steps
            first_audio, sr, _, first_trajectory = infer_single_process(
                ref_audio_preprocessed, ref_text_preprocessed, text, ema_model, vocoder, 
                mel_spec_type=mel_spec_type, 
                nfe_step=steps,  # Keep total steps same
                speed=speed,
                start_step=0,
                end_step=split_step,
                cfg_strength=2.0,
                seed=random.randint(0, 10000) #42  # Use a fixed seed for first stage
            )

            # TODO:
            # time stretch

            # Second stage - refine the first stage
            second_audio, sr, _, trajectory = infer_single_process(
                ref_audio_preprocessed, ref_text_preprocessed, text, ema_model_fine, vocoder,
                mel_spec_type=mel_spec_type,
                nfe_step=steps,  # Keep total steps same
                speed=speed,
                start_step=split_step,
                end_step=steps,
                initial_state=first_trajectory[-1],
                cfg_strength=2.0,
                seed=random.randint(0, 10000) #i + 100  # Different seed for each variant
            )

            # from 24khz to 16khz 
            audio = librosa.resample(normalize_audio(second_audio), orig_sr=sr, target_sr=16000)

            with torch.inference_mode():
                result = whisper_model.transcribe(audio, batch_size=1, language="ru")

            if len(result['segments']) > 0:
                out_text = result["segments"][0]['text'].lower()
            else:
                out_text = ""

            _, score, _ = find_best_match(text, [out_text])
            if score > max_score:
                max_score = score
                segment_audio = second_audio

            '''
            #print(j, f"score: {score}", out_text)
            
            segment_wave_path = Path(output_dir) / f"segment_audio_{i}_variant_{j}.mp3" 
            print(segment_wave_path)
            sf.write(segment_wave_path, second_audio, sr)
            '''

        audio_segments.append(segment_audio)
    
        # Save first stage audio
        out_dir = Path(output_dir) 
        out_dir.mkdir(exist_ok=True, parents=True)
        segment_wave_path = out_dir / f"segment_audio_{i}.mp3" 
        print(segment_wave_path)
        sf.write(segment_wave_path, segment_audio, sr)

        ref_text = text
        ref_audio = str(segment_wave_path)
        #ref_audio_preprocessed, ref_text_preprocessed = preprocess_ref_audio_text( ref_audio, ref_text)
        
    # Combine all segments and save final output
    combined_audio = np.concatenate(audio_segments)
    sf.write(final_path, combined_audio, sr)


if __name__ == '__main__':
    # test 
    gen_text = "zima-holoda, odinokie doma. morja, goroda — vsjo kak budto izo l'da. no skoro vesna, sneg rastaet i togda. za beloj stenoj my ostanemsja s toboj.   vstretilis' my posredine zimy. no drug druga s toboj ne mogli ne uznat'"

    # Select ref section
    ref_track_name = list(base_ref_tracks.keys())[0]

    final_path = Path(output_dir) / "out.mp3"
 
    inference_by_segments(ref_track_name, gen_text, final_path)
