from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
import numpy as np

def find_best_match(input_text, output_texts):
    """
    Finds the best matching text from output_texts compared to input_text
    using multiple similarity metrics.
    
    Parameters:
    input_text (str): The original text
    output_texts (list): List of strings to compare against input_text
    
    Returns:
    tuple: (best_matching_text, similarity_score, index)
    """
    def normalize_text(text):
        """Normalize text for comparison"""
        return text.lower().strip()
    
    def get_similarity_score(text1, text2):
        """
        Calculate similarity score using multiple metrics
        Returns a score between 0 and 1 (1 being most similar)
        """
        # Normalize texts
        text1 = normalize_text(text1)
        text2 = normalize_text(text2)
        
        # Sequence Matcher similarity (0-1)
        sequence_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # Levenshtein distance normalized (convert to similarity)
        max_len = max(len(text1), len(text2))
        levenshtein_sim = 1 - (levenshtein_distance(text1, text2) / max_len)
        
        # Word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Combine scores (you can adjust weights if needed)
        weights = [0.4, 0.4, 0.2]  # Sequence, Levenshtein, Word overlap
        combined_score = np.average([sequence_sim, levenshtein_sim, word_overlap], 
                                  weights=weights)
        
        return combined_score
    
    # Calculate scores for all outputs
    scores = [(text, get_similarity_score(input_text, text), i) 
             for i, text in enumerate(output_texts)]
    
    # Find the best match
    best_match = max(scores, key=lambda x: x[1])
    return best_match  # Returns (text, score, index)

if __name__ == '__main__':
    # test 
    res = find_best_match("hello world", ["hell", "he1llo", "world", "worl", "worl hello"])
    print(res)