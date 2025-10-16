#!/usr/bin/env python3
"""
Baseline VAD Scorer for Test Sentences

This script scores sentences using the NRC-VAD lexicon v2.1 by:
1. Loading the English unigrams lexicon
2. Processing each sentence by tokenizing and looking up word scores
3. Averaging the VAD scores for all found words in each sentence
4. Outputting results to a CSV file

Author: AraVAD Project
"""

import pandas as pd
import re
import csv
from pathlib import Path

def load_vad_lexicon(lexicon_path):
    """
    Load the NRC-VAD lexicon from the unigrams file.
    
    Args:
        lexicon_path (str): Path to the unigrams VAD lexicon file
        
    Returns:
        dict: Dictionary mapping words to their VAD scores
    """
    vad_dict = {}
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as file:
            # Skip header line
            next(file)
            
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    term, valence, arousal, dominance = parts
                    vad_dict[term.lower()] = {
                        'valence': float(valence),
                        'arousal': float(arousal),
                        'dominance': float(dominance)
                    }
    except FileNotFoundError:
        print(f"Error: Could not find lexicon file at {lexicon_path}")
        return None
    except Exception as e:
        print(f"Error loading lexicon: {e}")
        return None
        
    print(f"Loaded {len(vad_dict)} terms from VAD lexicon")
    return vad_dict

def tokenize_sentence(sentence):
    """
    Simple tokenization - split on whitespace and remove punctuation.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        list: List of cleaned tokens
    """
    # Convert to lowercase and remove punctuation
    sentence = sentence.lower()
    # Keep only letters, numbers, and spaces
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    # Split on whitespace and filter empty strings
    tokens = [token.strip() for token in sentence.split() if token.strip()]
    return tokens

def score_sentence(sentence, vad_dict):
    """
    Score a sentence by averaging VAD scores of found words.
    
    Args:
        sentence (str): Input sentence
        vad_dict (dict): VAD lexicon dictionary
        
    Returns:
        tuple: (valence_score, arousal_score, dominance_score, found_words_count)
    """
    tokens = tokenize_sentence(sentence)
    
    valence_scores = []
    arousal_scores = []
    dominance_scores = []
    
    for token in tokens:
        if token in vad_dict:
            valence_scores.append(vad_dict[token]['valence'])
            arousal_scores.append(vad_dict[token]['arousal'])
            dominance_scores.append(vad_dict[token]['dominance'])
    
    # Calculate averages if we found any words
    if len(valence_scores) > 0:
        avg_valence = sum(valence_scores) / len(valence_scores)
        avg_arousal = sum(arousal_scores) / len(arousal_scores)
        avg_dominance = sum(dominance_scores) / len(dominance_scores)
        found_words = len(valence_scores)
    else:
        # No words found in lexicon - assign neutral scores
        avg_valence = 0.0
        avg_arousal = 0.0
        avg_dominance = 0.0
        found_words = 0
    
    return avg_valence, avg_arousal, avg_dominance, found_words

def main():
    """Main function to process test sentences and create output CSV."""
    
    # Define paths
    base_path = Path("/Volumes/The Storage!/AraVAD")
    lexicon_path = base_path / "Datasets/Raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
    test_sentences_path = base_path / "Datasets/test_sentences.csv"
    output_path = base_path / "Datasets/baseline_vad_scores.csv"
    
    print("Loading NRC-VAD lexicon...")
    vad_dict = load_vad_lexicon(lexicon_path)
    
    if vad_dict is None:
        print("Failed to load lexicon. Exiting.")
        return
    
    print("Loading test sentences...")
    try:
        # Read test sentences
        test_df = pd.read_csv(test_sentences_path)
        sentences = test_df['Sentence'].tolist()
        print(f"Loaded {len(sentences)} test sentences")
    except Exception as e:
        print(f"Error loading test sentences: {e}")
        return
    
    print("Scoring sentences...")
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        valence, arousal, dominance, found_words = score_sentence(sentence, vad_dict)
        
        results.append({
            'sentence': sentence,
            'valence': round(valence, 4),
            'arousal': round(arousal, 4),
            'dominance': round(dominance, 4)
        })
        
        # Print progress every 20 sentences
        if i % 20 == 0:
            print(f"Processed {i}/{len(sentences)} sentences")
    
    print("Saving results to CSV...")
    try:
        # Create output CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sentence', 'valence', 'arousal', 'dominance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"Results saved to: {output_path}")
        print(f"Processed {len(results)} sentences successfully")
        
        # Print some statistics
        valences = [r['valence'] for r in results]
        arousals = [r['arousal'] for r in results]
        dominances = [r['dominance'] for r in results]
        
        print("\nBaseline VAD Score Statistics:")
        print(f"Valence - Mean: {sum(valences)/len(valences):.4f}, Min: {min(valences):.4f}, Max: {max(valences):.4f}")
        print(f"Arousal - Mean: {sum(arousals)/len(arousals):.4f}, Min: {min(arousals):.4f}, Max: {max(arousals):.4f}")
        print(f"Dominance - Mean: {sum(dominances)/len(dominances):.4f}, Min: {min(dominances):.4f}, Max: {max(dominances):.4f}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
