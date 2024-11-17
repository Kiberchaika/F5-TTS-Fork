# Поиск треков для перебора
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re
import json
import os
from dataclasses import dataclass
import unicodedata

@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    y_position: float
    height: float

@dataclass
class SectionData:
    words: List[WordInfo]
    words_str: str

@dataclass
class MirData:
    bpm: float
    path: str
    beats: List[float]
    downbeats: List[float]
    beat_positions: List[int]
    segments: List[Dict[str, Any]]
    moods: List[str]
    genres: List[str]

def normalize_filename(filename: str) -> str:
    """
    Normalize Unicode string to NFC form (composed form)
    """
    return unicodedata.normalize('NFC', filename)

def parse_mir_json(json_path: str) -> Optional[MirData]:
    """
    Parse the .mir.json file
    
    Args:
        json_path: Path to the .mir.json file
        
    Returns:
        MirData object or None if parsing fails
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return MirData(
                bpm=data.get('bpm', 0.0),
                path=data.get('path', ''),
                beats=data.get('beats', []),
                downbeats=data.get('downbeats', []),
                beat_positions=data.get('beat_positions', []),
                segments=data.get('segments', []),
                moods=data.get('moods', []),
                genres=data.get('genres', [])
            )
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None
    
def check_file_exists(file_path: Path) -> bool:
    """
    More robust file existence check that handles spaces and special characters.
    """
    return os.path.exists(str(file_path))

# Version 2: Using glob to find the exact file
def find_mir_file(base_path: Path, base_name: str) -> Optional[Path]:
    """
    Find .mir.json file using direct file construction.
    """
    try:
        # Simply construct the expected file path without escaping
        mir_file = base_path / f"{base_name}.mir.json"
        if os.path.isfile(str(mir_file)):
            return mir_file
            
        # If not found, try listing directory contents to find a match
        for file in os.listdir(base_path):
            if file == f"{base_name}.mir.json":
                return base_path / file
                
        return None
    except Exception as e:
        print(f"Error finding mir file for {base_name}: {e}")
        return None

def read_caption_file(file_path: str) -> str:
    """
    Read caption file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading caption file {file_path}: {e}")
        return ""


def parse_sentence(words: List[WordInfo], max_chars: int = 100) -> tuple[str, float, List[WordInfo]]:
    """
    Parse words into a sentence until reaching max_chars limit.
    
    Args:
        words: List of WordInfo objects
        max_chars: Maximum characters allowed in the sentence
    
    Returns:
        tuple: (constructed sentence, end time of last word, remaining words)
    """
    if not words:
        return "", 0.0, []
    
    current_chars = 0
    sentence_words = []
    remaining_words = []
    last_end_time = 0.0
    
    for i, word_info in enumerate(words):
        # Calculate length with added word (including space)
        word_length = len(word_info.word) + (1 if sentence_words else 0)
        
        if current_chars + word_length <= max_chars:
            sentence_words.append(word_info)
            current_chars += word_length
            last_end_time = word_info.end
        else:
            remaining_words = words[i:]
            break
    else:  # If we process all words without breaking
        remaining_words = []
    
    # Construct the sentence with proper spacing
    sentence = " ".join(word.word for word in sentence_words)
    
    return sentence, last_end_time


def process_music_ref_dataset(dataset_path: str, cache_filename: str = "ref_data_cache.json", rewrite_cache: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Enhanced version that reads caption files and section words
    """
    # Check for existing cache
    cache_path = Path(cache_filename)
    if cache_path.exists() and not rewrite_cache:
        try:
            print(f"Loading cached data from {cache_filename}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file: {e}, rebuilding...")
    
    result = {}
    root_path = Path(dataset_path)
    skipped_tracks = []
    
    print(f"Processing dataset at {dataset_path}")
    
    for caption_file in root_path.rglob("*_caption.txt"):
        try:
            # Caption and base name processing
            caption_name = normalize_filename(caption_file.name)
            base_name = normalize_filename(caption_name.rsplit('_caption.txt', 1)[0])
            base_path = caption_file.parent
            
            # Read caption content
            caption_content = read_caption_file(str(caption_file))
            
            # MIR file finding and parsing remains the same...
            mir_filename = normalize_filename(f"{base_name}.mir.json")
            mir_file = None
            
            for file in os.listdir(base_path):
                if normalize_filename(file) == mir_filename:
                    mir_file = base_path / file
                    break
            
            if not mir_file:
                print(f"Debug: Failed to find .mir.json for {base_name}")
                skipped_tracks.append((base_name, "Missing .mir.json file"))
                continue
                
            try:
                mir_data = parse_mir_json(str(mir_file))
                if not mir_data:
                    skipped_tracks.append((base_name, "Failed to parse .mir.json"))
                    continue
            except Exception as e:
                print(f"Debug: Error processing {base_name}: {str(e)}")
                skipped_tracks.append((base_name, f"Error processing .mir.json: {str(e)}"))
                continue
                
            # Section processing
            section_pattern = re.compile(rf"{re.escape(normalize_filename(base_name))}_vocals_stretched_120bpm_section(\d+)")
            sections_dict = {}
            
            # Find all files that match the base pattern
            for file_path in base_path.glob(f"{base_name}_vocals_stretched_120bpm_section*"):
                file_name = normalize_filename(file_path.name)
                match = section_pattern.match(file_name)
                if match:
                    section_num = int(match.group(1))
                    file_ext = file_path.suffix
                    
                    if section_num not in sections_dict:
                        sections_dict[section_num] = {"json": None, "mp3": None}
                    
                    if file_ext == ".json":
                        sections_dict[section_num]["json"] = str(file_path)
                    elif file_ext == ".mp3":
                        sections_dict[section_num]["mp3"] = str(file_path)
            
            # Process sections and include words
            sections = []
            for section_num in sorted(sections_dict.keys()):
                section = sections_dict[section_num]
                if section["json"] and section["mp3"]:
                    # Parse section JSON and get words
                    section_data = parse_section_json(section["json"])
                    sentence, last_end_time = parse_sentence(section_data.words, 100)
                    
                    sections.append({
                        "json_path": section["json"],
                        "mp3_path": section["mp3"],
                        "words": sentence, #section_data.words_str,
                        "end_time": last_end_time * 120 / mir_data.bpm,
                    })
            
            if not sections:
                skipped_tracks.append((base_name, "No valid section pairs found"))
                continue

            # Initialize dictionary for this base_name with new structure
            result[base_name] = {
                "caption": caption_content,  # Now contains file content instead of path
                "mir": str(mir_file),
                "mir_data": {
                    "bpm": mir_data.bpm,
                    "path": mir_data.path,
                    "moods": mir_data.moods,
                    "genres": mir_data.genres
                },
                "sections": sections  # Now contains dictionary with paths and words
            }

        except Exception as e:
            print(f"Error processing {caption_file}: {e}")
            continue

    # Skipped tracks reporting and cache saving remains the same...
    if skipped_tracks:
        print("\nSkipped tracks report:")
        for track, reason in skipped_tracks:
            print(f"- {track}: {reason}")
        print(f"\nTotal skipped tracks: {len(skipped_tracks)}")

    try:
        print(f"\nSaving cache to {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save cache file: {e}")

    return result

def get_dict_key_by_index(data_dict: Dict, index: int) -> str:
    """
    Get dictionary key by its index.
    """
    try:
        return list(data_dict.keys())[index]
    except IndexError:
        raise IndexError(f"Index {index} is out of range. Dictionary has {len(data_dict)} keys")

def parse_section_json(json_path: str) -> SectionData:
    """
    Parse the section JSON file and extract words information.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract words data
        words_data = data.get('words', [])
        words = [WordInfo(**word_info) for word_info in words_data]
        
        # Create concatenated string
        words_str = ' '.join(word_info.word for word_info in words)
        
        return SectionData(words=words, words_str=words_str)
    
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return SectionData(words=[], words_str="")
    

# Usage example

# ref_track_name = list(base_ref_tracks.keys())[515]
# ref_track = base_ref_tracks[ref_track_name]
# print(ref_track_name)
# print(ref_track['mir'])
# print(ref_track['caption'])
# print(ref_track['mir_data']['bpm'])
# print(f"how many sections: {len(ref_track['sections'])}")
# print(ref_track['sections'][0]['words'])
# print(ref_track['sections'][0]['mp3_path'])