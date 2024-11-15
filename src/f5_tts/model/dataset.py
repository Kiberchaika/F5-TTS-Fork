import json
import random
from importlib.resources import files

from audio_augment import augment_audio

from utils import (
    get_words_in_timerange,
    clean_string,
    get_timestamp,
    find_matching_filename
)
from verse_chorus_vocal_cutter import (
    get_verses_count,
    get_choruses_count,
    get_vocal_cut_from_chorus,
    get_vocal_cut_from_verse
)

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

from typing import List, Dict
import os
import json
import cyrtranslit
 
class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = row["duration"]

        if duration > 50 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            data_path = "/root/Music_stretched_dataset"
            audio, source_sample_rate = torchaudio.load(os.path.join(data_path, audio_path))
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

            text = cyrtranslit.to_latin(text, "ru").lower()
            text = text.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip().lower()  
            text = convert_char_to_pinyin([text], polyphone=True)[0]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class RussianSingingDataset2(Dataset):
    def __init__(
        self,
        base_dir="/media/k4_nas/Datasets/Music_RU/Vocal_Dereverb",
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        
        self.use_augmentation = True

        # Find all MP3 files recursively
        self.json_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.mir.json'):
                    self.json_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.json_files)} files")
        
        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        # Return approximate frame length for a 25-second audio
        return (self.max_duration * self.target_sample_rate / self.hop_length)

    def __len__(self):
        return len(self.json_files) * 4 # (200 * 60 * 60 // self.max_duration) # size of dataset

    def __getitem__(self, index):
        # Get random file regardless of index
        
        while True:
            json_path = self.json_files[index % len(self.json_files)] #random.choice(self.json_files)

            total_duration = 0

            audio_path = find_matching_filename(json_path.replace('.mir.json', '_vocals_noreverb.mp3'))
            input_audio, source_sample_rate = torchaudio.load(audio_path)

            if input_audio.shape[0] > 1:
                input_audio = torch.mean(input_audio, dim=0, keepdim=True)
            
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    source_sample_rate, 
                    self.target_sample_rate
                )
                input_audio = resampler(input_audio)

            # Get path to song layout
            audio = None

            ref_audio_mir = json_path.replace('.mir.json', '') 
            if random.random() > 0.5:
                verse_count = get_verses_count(ref_audio_mir)
                if verse_count:
                    i = random.randint(0, verse_count - 1)
                    success, start_time, end_time = get_vocal_cut_from_verse(ref_audio_mir, "", i, False)
                    if success:
                        audio = (input_audio[0][int(start_time * self.target_sample_rate) : int(end_time * self.target_sample_rate)]) 
                        total_duration = end_time - start_time
            else:
                chorus_count = get_choruses_count(ref_audio_mir)
                if chorus_count:
                    i = random.randint(0, chorus_count - 1)
                    success, start_time, end_time = get_vocal_cut_from_chorus(ref_audio_mir, "", i, False)
                    if success:
                        audio = (input_audio[0][int(start_time * self.target_sample_rate) : int(end_time * self.target_sample_rate)]) 
                        total_duration = end_time - start_time
 
            if audio is None or total_duration > 50:
                index += 1
                continue

            # Get lyrics
            ref_lyrics = find_matching_filename(json_path.replace('.mir.json', "_vocals_noreverb.json"))
            text = get_words_in_timerange(ref_lyrics, start_time, end_time)
                        
            text = cyrtranslit.to_latin(text, "ru").lower()
            text = text.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip().lower()  
            if text == '':
                continue

            if len(text) < 10:
                continue

            text = convert_char_to_pinyin([text], polyphone=True)[0]
            audio = audio.unsqueeze(0)

            if self.use_augmentation:
                audio = augment_audio(audio, self.target_sample_rate)

            break
    
        if self.preprocessed_mel:
            raise NotImplementedError("Preprocessed mel not supported in this version")
        
        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
        
        return dict(
            mel_spec=mel_spec,
            text=text,
        )
   
class RussianSingingDataset(Dataset):
    def __init__(
        self,
        base_dir="/media/k4_nas/Datasets/Music_RU/Lyrics",
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        
        self.min_duration = 5
        self.max_duration = 25 
        self.use_augmentation = True

        # Find all MP3 files recursively
        self.json_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    self.json_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.json_files)} files")
        
        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        # Return approximate frame length for a 25-second audio
        return (self.max_duration * self.target_sample_rate / self.hop_length)

    def __len__(self):
        return len(self.json_files) * 4 # (200 * 60 * 60 // self.max_duration) # size of dataset

    def __getitem__(self, index):
        # Get random file regardless of index
        
        while True:
            json_path = self.json_files[index % len(self.json_files)] #random.choice(self.json_files)

            total_duration = 0
            
            # Parse the JSON data
            filepath_json = open(json_path, 'r', encoding='utf-8') 
            segments = json.load(filepath_json)

            for segment in segments:
                duration = segment['end'] - segment['start']
                total_duration += duration

            # If total duration is greater than target duration, break
            if total_duration < self.min_duration:
                index += 1
                continue

            # get filepath from filepath_json with replace .json to .mp3
            audio_path = json_path.replace('Lyrics', 'Vocal_Dereverb').replace('.json', '.mp3')
            input_audio, source_sample_rate = torchaudio.load(audio_path)
            
            if input_audio.shape[0] > 1:
                input_audio = torch.mean(input_audio, dim=0, keepdim=True)
            
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    source_sample_rate, 
                    self.target_sample_rate
                )
                input_audio = resampler(input_audio)


            # Get random start index
            start_idx = random.randint(0, len(segments) - 1)
            
            # Collect segments until we reach target duration
            collected_segments = []
            current_duration = 0
            idx = start_idx
            
            while current_duration < self.max_duration and idx < len(segments):
                segment = segments[idx]
                duration = segment['end'] - segment['start']
                if current_duration + duration > self.max_duration:
                    break
                current_duration += duration
                collected_segments.append(segment)
                idx += 1

            if current_duration > self.max_duration or current_duration < self.min_duration:
                index += 1
                continue

            texts = [] 
            audio = []
            for i in range(start_idx, idx):
                segment = segments[i]

                text = cyrtranslit.to_latin(segment['text'], "ru").lower()
                texts.append(text)

                # todo: если пропуск больше 5сек то пропускаем сэмпл

                # get segment from audio segment['start'] in ms to audio position
                audio.append(input_audio[0][int(segment['start'] * self.target_sample_rate) : int(segment['end'] * self.target_sample_rate)]) 

            text = ' '.join(texts).replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip().lower()   
            if len(text) < 10:
                index += 1
                continue
            
            text = convert_char_to_pinyin([text], polyphone=True)[0]
        
            audio = torch.cat(audio, dim=0)
            if audio.shape[0] < self.max_duration * self.target_sample_rate:
                audio = torch.cat([audio, torch.zeros(self.max_duration * self.target_sample_rate - audio.shape[0])], dim=0)
            audio = audio.unsqueeze(0)

            if self.use_augmentation:
                audio = augment_audio(audio, self.target_sample_rate)

            # save audio as out.mp3
            #torchaudio.save(f'/home/k4/Python/F5-TTS/out.mp3', audio, self.target_sample_rate)

            break

    
        if self.preprocessed_mel:
            raise NotImplementedError("Preprocessed mel not supported in this version")
        
        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
        
        return dict(
            mel_spec=mel_spec,
            text=text,
        )
       
# Dynamic Batch Sampler

class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = "/root/Vocal_Dereverb_Prepared" # str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "RussianSingingDataset": 
        train_dataset = RussianSingingDataset(
            
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        dataset_name = "/root/Vocal_Dereverb_Prepared"
        preprocessed_mel = False

        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )

