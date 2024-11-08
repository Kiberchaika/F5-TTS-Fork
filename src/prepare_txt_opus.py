import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter
import torchaudio
import argparse
from importlib.resources import files

from f5_tts.model.utils import (
    convert_char_to_pinyin,
)

PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")

def find_opus_txt_pairs(root_dir):
    """Find all matching .opus and .txt files in the directory structure."""
    root_path = Path(root_dir)
    opus_files = list(root_path.rglob("*.opus"))
    pairs = []
    
    for opus_file in opus_files:
        txt_file = opus_file.with_suffix('.txt')
        if txt_file.exists():
            pairs.append((opus_file, txt_file))
    
    return pairs

def get_audio_duration(audio_path):
    """Get duration of an opus audio file."""
    audio, sample_rate = torchaudio.load(audio_path)
    return audio.shape[1] / sample_rate

def prepare_dataset(input_dir):
    """Prepare dataset from opus/txt pairs."""
    file_pairs = find_opus_txt_pairs(input_dir)
    
    sub_result, durations = [], []
    vocab_set = set()
    polyphone = True
    
    for opus_path, txt_path in tqdm(file_pairs, desc="Processing audio files"):
        try:
            # Read text content
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Get audio duration
            audio_duration = get_audio_duration(opus_path)
            
            # Convert text to pinyin
            text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
            
            # Store results
            sub_result.append({
                "audio_path": opus_path.as_posix(),
                "text": text,
                "duration": audio_duration
            })
            durations.append(audio_duration)
            vocab_set.update(list(text))
            
        except Exception as e:
            print(f"Error processing {opus_path}: {str(e)}")
            continue
    
    return sub_result, durations, vocab_set

def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune):
    """Save the prepared dataset to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    # Save to arrow format
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=1) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocabulary
    voca_out_path = out_dir / "vocab.txt"
    if is_finetune:
        file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        if not Path(file_vocab_finetune).exists():
            raise FileNotFoundError(f"Pretrained vocab file not found: {file_vocab_finetune}")
        os.system(f"cp {file_vocab_finetune} {voca_out_path}")
    else:
        with open(voca_out_path, "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}:")
    print(f"Sample count: {len(result)}")
    print(f"Vocab size: {len(text_vocab_set)}")
    print(f"Total duration: {sum(duration_list)/3600:.2f} hours")

def prepare_and_save_set(inp_dir, out_dir, is_finetune: bool = True):
    """Main function to prepare and save the dataset."""
    if is_finetune:
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    sub_result, durations, vocab_set = prepare_dataset(inp_dir)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune)

def cli():
    parser = argparse.ArgumentParser(description="Prepare and save opus/txt dataset.")
    parser.add_argument("--inp_dir", default="/mnt/datasets/radio_2", type=str, help="Input directory containing the opus/txt files.")
    parser.add_argument("--out_dir", default="/mnt/datasets/radio_2_processed", type=str, help="Output directory to save the prepared data.")
    parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")

    args = parser.parse_args()
    prepare_and_save_set(args.inp_dir, args.out_dir, is_finetune=not args.pretrain)

if __name__ == "__main__":
    cli()