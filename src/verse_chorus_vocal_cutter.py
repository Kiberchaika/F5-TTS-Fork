import json
from pathlib import Path
import logging
import torchaudio
import torch
from typing import Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_base_path_and_json(input_file_path: str) -> Tuple[str, Path]:
    """Helper function to get base name and JSON path."""
    path = Path(input_file_path)
    base_name = path.name
    
    # Remove all extensions
    # while '.' in base_name:
    #     base_name = Path(base_name).stem
        
    # Remove instrumental/vocals suffixes if present
    if base_name.endswith('_instrumental'):
        base_name = base_name[:-12]
    elif base_name.endswith('_vocals'):
        base_name = base_name[:-7]
    elif base_name.endswith('_vocals_noreverb'):
        base_name = base_name[:-15]
        
    json_path = path.parent / f"{base_name}.mir.json"
    return base_name, json_path

def load_json_data(json_path: Path) -> Optional[dict]:
    """Helper function to load JSON data."""
    try:
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return None
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return None

def get_verses_count(input_file_path: str) -> Optional[int]:
    """
    Get the number of verses in the song.
    
    Args:
        input_file_path: Path to any file related to the song 
                        (instrumental, vocals, or .mir.json)
    
    Returns:
        Number of verses or None if error occurs
    """
    _, json_path = get_base_path_and_json(input_file_path)
    data = load_json_data(json_path)
    
    if not data:
        return None
        
    verses = [seg for seg in data['segments'] if seg['label'].lower() == 'verse']
    count = len(verses)
    logger.info(f"Found {count} verses in the song")
    return count

def get_choruses_count(input_file_path: str) -> Optional[int]:
    """
    Get the number of choruses in the song.
    
    Args:
        input_file_path: Path to any file related to the song
                        (instrumental, vocals, or .mir.json)
    
    Returns:
        Number of choruses or None if error occurs
    """
    _, json_path = get_base_path_and_json(input_file_path)
    data = load_json_data(json_path)
    
    if not data:
        return None
        
    choruses = [seg for seg in data['segments'] if seg['label'].lower() == 'chorus']
    count = len(choruses)
    logger.info(f"Found {count} choruses in the song")
    return count

def get_vocal_cut_from_chorus(input_file_path: str, 
                            output_file_path: str, 
                            chorus_index: int, need_to_save: bool = False) -> bool:
    """
    Cut a specific chorus from the vocals_noreverb file and save it.
    
    Args:
        input_file_path: Path to any file related to the song
        output_file_path: Where to save the cut chorus
        chorus_index: Which chorus to cut (0-based index)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get paths
        base_name, json_path = get_base_path_and_json(input_file_path)
        vocals_path = Path(input_file_path).parent / f"{base_name}_vocals_noreverb.opus"

        if need_to_save:
            # Check files exist
            if not vocals_path.exists():
                logger.error(f"Vocals file not found: {vocals_path}")
                return False, None, None
            
        # Load JSON data
        data = load_json_data(json_path)
        if not data:
            return False, None, None
            
        # Get choruses
        choruses = [seg for seg in data['segments'] if seg['label'].lower() == 'chorus']
        if not choruses:
            logger.error("No choruses found in the song")
            return False, None, None
            
        # Validate chorus index
        if chorus_index >= len(choruses):
            logger.error(f"Chorus index {chorus_index} out of range (found {len(choruses)} choruses)")
            return False, None, None
            
        # Get timing for requested chorus
        chorus = choruses[chorus_index]
        start_time = chorus['start']
        end_time = chorus['end']
        logger.info(f"Cutting chorus {chorus_index} from {start_time:.2f}s to {end_time:.2f}s")

        if not need_to_save:
            return True, start_time, end_time
          
        # Load audio
        waveform, sample_rate = torchaudio.load(str(vocals_path))
        
        # Convert times to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Cut the chorus
        chorus_audio = waveform[:, start_sample:end_sample]
        
        # Save
        torchaudio.save(output_file_path, chorus_audio, sample_rate)
        logger.info(f"Saved chorus to {output_file_path}")
        
        return True, start_time, end_time
        
    except Exception as e:
        logger.error(f"Error processing vocal cut: {e}")
        return False, None, None
    
def get_vocal_cut_from_verse(input_file_path: str, 
                            output_file_path: str, 
                            verse_index: int, need_to_save: bool = False) -> bool:
    """
    Cut a specific chorus from the vocals_noreverb file and save it.
    
    Args:
        input_file_path: Path to any file related to the song
        output_file_path: Where to save the cut chorus
        verse_index: Which verse to cut (0-based index)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get paths
        base_name, json_path = get_base_path_and_json(input_file_path)
        vocals_path = Path(input_file_path).parent / f"{base_name}_vocals_noreverb.opus"
        
        if need_to_save:
            # Check files exist
            if not vocals_path.exists():
                logger.error(f"Vocals file not found: {vocals_path}")
                return False, None, None
            
        # Load JSON data
        data = load_json_data(json_path)
        if not data:
            return False, None, None
            
        # Get verses
        verses = [seg for seg in data['segments'] if seg['label'].lower() == 'verse']
        if not verses:
            logger.error("No verses found in the song")
            return False, None, None
            
        # Validate verse index
        if verse_index >= len(verses):
            logger.error(f"Verse index {verse_index} out of range (found {len(verses)} verses)")
            return False, None, None
            
        # Get timing for requested chorus
        verse = verses[verse_index]
        start_time = verse['start']
        end_time = verse['end']
        logger.info(f"Cutting verse {verse_index} from {start_time:.2f}s to {end_time:.2f}s")

        if not need_to_save:
            return True, start_time, end_time
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(vocals_path))
        
        # Convert times to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Cut the chorus
        verse_audio = waveform[:, start_sample:end_sample]
        
        # Save
        torchaudio.save(output_file_path, verse_audio, sample_rate)
        logger.info(f"Saved verse to {output_file_path}")
        
        return True, start_time, end_time
        
    except Exception as e:
        logger.error(f"Error processing vocal cut: {e}")
        return False, None, None