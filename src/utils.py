import os
from datetime import datetime
import os
import unicodedata
import re
import json

def get_words_in_timerange(filename, start_time, end_time):
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    words = []
    
    for segment in json_data:
        for word_info in segment.get('words', []):
            if not ('start' in word_info and 'end' in word_info):
                continue
                
            word_start = word_info['start']
            word_end = word_info['end']
            
            if word_start >= start_time and word_end <= end_time:
                words.append(word_info['word'])
    
    return ' '.join(words)

def clean_string(text: str):
    # Remove BOM and other unicode control characters
    text = text.replace('\ufeff', '')
    
    # Remove other unicode control characters
    text = ''.join(char for char in text if ord(char) >= 32)
    
    return text

def get_timestamp(format_type: str = 'default'):
    """
    Generate timestamp string for filenames.
    
    Args:
        format_type (str): Type of timestamp format to use
            'default': '20241110_153021' (YYYYMMDD_HHMMSS)
            'simple': '153021' (HHMMSS)
            'detailed': '20241110_153021_123' (YYYYMMDD_HHMMSS_MS)
            'compact': '20241110153021' (YYYYMMDDHHMMSS)
            'readable': '2024-11-10_15-30-21' (YYYY-MM-DD_HH-MM-SS)
            
    Returns:
        str: Formatted timestamp string
    """
    formats = {
        'default': '%Y%m%d_%H%M%S',
        'simple': '%H%M%S',
        'detailed': '%Y%m%d_%H%M%S_%f',
        'compact': '%Y%m%d%H%M%S',
        'readable': '%Y-%m-%d_%H-%M-%S'
    }
    
    timestamp_format = formats.get(format_type, formats['default'])
    return datetime.now().strftime(timestamp_format)

def find_matching_filename(filepath):
    """
    Find a matching filename in the given directory path, ignoring unicode characters
    and hidden symbols.
    
    Args:
        filepath (str): Full path to the file
    
    Returns:
        str: Path to the matching file if found, None if no match
    """
    def normalize_filename(filename):
        # Convert to NFKD normalized form and encode as ASCII, ignoring non-ASCII
        normalized = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
        
        # Remove hidden characters and symbols
        cleaned = re.sub(r'[\u200B-\u200D\uFEFF]', '', normalized)  # Remove zero-width chars
        cleaned = re.sub(r'[^\w\s.-]', '', cleaned)  # Keep only alphanumeric, dots, hyphens
        cleaned = cleaned.lower().strip()  # Convert to lowercase and strip whitespace
        return cleaned
    
    try:
        # Split filepath into directory and filename
        directory = os.path.dirname(filepath)
        target_filename = os.path.basename(filepath)
        
        # If directory is empty, use current directory
        if not directory:
            directory = '.'
            
        # Normalize the target filename
        normalized_target = normalize_filename(target_filename)
        
        # Scan directory for matching files
        for filename in os.listdir(directory):
            if normalize_filename(filename) == normalized_target:
                return os.path.join(directory, filename)
        
        return None
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def split_text(text, max_length=132):
    """
    Split text into chunks of maximum length while preserving whole words
    and trying to break at sentence boundaries when possible.
    
    Args:
        text (str): Input text to split
        max_length (int): Maximum length of each chunk (default: 132)
    
    Returns:
        list: List of text chunks
    """
    # First split into sentences
    sentences = []
    current = []
    
    # Basic sentence splitting on ., !, ?
    for char in text:
        current.append(char)
        if char in '.!?' and len(current) > 0:
            sentences.append(''.join(current).strip())
            current = []
    
    # Add any remaining text as a sentence
    if current:
        sentences.append(''.join(current).strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # If single sentence is longer than max_length, need to split on words
        if len(sentence) > max_length:
            words = sentence.split()
            for word in words:
                # If adding this word would exceed max_length
                if current_length + len(word) + 1 > max_length:
                    # Store current chunk and start new one
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1  # +1 for space
        
        # If whole sentence fits within remainder of current chunk
        elif current_length + len(sentence) <= max_length:
            if current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
            else:
                current_chunk = [sentence]
                current_length = len(sentence)
        
        # If sentence doesn't fit, store current chunk and start new with this sentence
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    # Add final chunk if exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks