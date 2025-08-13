#!/usr/bin/env python3
"""
Script to filter loogle_test.csv for questions with short answers and save the top k results.
"""

import pandas as pd
from pathlib import Path
import argparse


def filter_short_answers_and_save(csv_path, output_path, k):
    """
    Filter dataset for questions with less than 10 words in answer, take top k, and save.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path for the output CSV file
        k: Number of questions to take after filtering
    """
    print(f"Filtering questions with short answers from {csv_path}...")
    
    # Read the entire CSV file
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} total rows from the dataset")
    
    # Filter for questions with less than 10 words in answer
    df['answer_word_count'] = df['answer'].str.split().str.len()
    short_answer_df = df[df['answer_word_count'] < 10].copy()
    print(f"Found {len(short_answer_df)} questions with answers less than 10 words")
    
    # Take the first k questions
    if len(short_answer_df) < k:
        print(f"Warning: Only {len(short_answer_df)} questions have short answers, using all of them")
        selected_df = short_answer_df
    else:
        selected_df = short_answer_df.head(k)
    
    print(f"Selected {len(selected_df)} questions")
    
    # Remove the temporary word count column
    selected_df = selected_df.drop('answer_word_count', axis=1)
    
    # Save the filtered dataset
    selected_df.to_csv(output_path, index=False)
    print(f"Filtered dataset saved at: {output_path}")
    print(f"Total questions saved: {len(selected_df)}")


def main():
    parser = argparse.ArgumentParser(description='Filter dataset for questions with short answers and save top k results')
    parser.add_argument('--input', default='data/loogle_test.csv', 
                       help='Input CSV file path (default: data/loogle_test.csv)')
    parser.add_argument('--output', default='data/loogle_questions_with_short_answers.csv',
                       help='Output CSV file path (default: data/loogle_questions_with_short_answers.csv)')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of questions with short answers to save (default: 20)')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    filter_short_answers_and_save(args.input, args.output, args.k)


if __name__ == "__main__":
    main()
