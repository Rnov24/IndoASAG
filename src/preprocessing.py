"""Preprocessing pipeline for the Rahutomo & Ari Roshinta (2018) Indonesian ASAG dataset.

Parses raw Excel sheets, cleans the text, computes normalized consensus scores,
and splits the data into stratified train/test partitions.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define paths relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset"
TOPICS = ["Lifestyle", "Olahraga", "Politik", "Teknologi"]


def parse_questions(filepath: Path) -> dict:
    """Extract question ID, question text, and reference answer from the 'Soal' sheet.

    Args:
        filepath: Path to the Excel file.

    Returns:
        Dict mapping question_id -> {"question_text": str, "reference_answer": str}
    """
    df = pd.read_excel(filepath, sheet_name="Soal", header=None)
    questions = {}
    
    # Raw layout: Row index 0 is header, data starts at index 1
    # Col 1: Question No, Col 2: Question Text, Col 3: Reference Answer
    for _, row in df.iterrows():
        q_num = row.iloc[1]
        if pd.isna(q_num):
            continue
        try:
            q_num = int(q_num)
        except (ValueError, TypeError):
            continue
            
        questions[q_num] = {
            "question_text": str(row.iloc[2]).strip(),
            "reference_answer": str(row.iloc[3]).strip(),
        }
    return questions


def parse_answers(filepath: Path, question_id: int) -> pd.DataFrame:
    """Extract student IDs, answers, and individual manual scores from the 'No.X' sheet.

    Args:
        filepath: Path to the Excel file.
        question_id: The question index (1 to 10).

    Returns:
        DataFrame containing student answers and scores.
    """
    df = pd.read_excel(filepath, sheet_name=f"No.{question_id}", header=None)
    
    # Locate header row containing "Siswa" or "No"
    header_row = None
    for idx, row in df.iterrows():
        vals = [str(v).strip() for v in row.values if pd.notna(v)]
        if "Siswa" in vals or "No" in vals:
            header_row = idx
            break
            
    if header_row is None:
        print(f"  [WARNING] Could not find header row in sheet No.{question_id}")
        return pd.DataFrame()
        
    data_rows = []
    # Data is situated after the header row
    for idx in range(header_row + 1, len(df)):
        row = df.iloc[idx]
        
        # Excel layout:
        # Col 2: student_id, Col 3: student_answer
        # Col 4: Score Rater 1, Col 5: Score Rater 2, Col 6: Score Rater 3
        # Col 13: Excel-computed Average
        student_id = row.iloc[2]
        answer = row.iloc[3]
        m1 = row.iloc[4]
        m2 = row.iloc[5]
        m3 = row.iloc[6]
        excel_avg = row.iloc[13] if len(row) > 13 else np.nan
        
        # Skip rows with missing identifiers or answers
        if pd.isna(student_id) or pd.isna(answer):
            continue
        student_answer = str(answer).strip()
        if student_answer == "" or student_answer.lower() == "nan":
            continue
            
        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan
                
        data_rows.append({
            "student_id": str(student_id).strip(),
            "student_answer": student_answer,
            "score_m1": to_float(m1),
            "score_m2": to_float(m2),
            "score_m3": to_float(m3),
            "score_excel_avg": to_float(excel_avg),
        })
        
    return pd.DataFrame(data_rows)


def build_master_dataset() -> pd.DataFrame:
    """Consolidate raw Excel sheets across all 4 topics into a single master DataFrame."""
    all_data = []
    
    for topic in TOPICS:
        filename = f"Analisis_Essay_Grading_{topic}.xlsx"
        filepath = RAW_DIR / filename
        
        if not filepath.exists():
            print(f"[WARNING] File not found: {filepath}. Skipping.")
            continue
            
        print(f"[Preprocess] Reading topic: {topic}")
        questions = parse_questions(filepath)
        
        for q_id, q_info in sorted(questions.items()):
            answers_df = parse_answers(filepath, q_id)
            if answers_df.empty:
                continue
                
            answers_df["topic"] = topic
            answers_df["question_id"] = q_id
            answers_df["question_text"] = q_info["question_text"]
            answers_df["reference_answer"] = q_info["reference_answer"]
            
            all_data.append(answers_df)
            
    if not all_data:
        raise FileNotFoundError(f"No Excel files parsed successfully in {RAW_DIR}")
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Calculate unified manual scores (using Excel average if present, otherwise average of available rater scores)
    score_cols = ["score_m1", "score_m2", "score_m3"]
    df["score_manual_avg"] = df["score_excel_avg"].fillna(df[score_cols].mean(axis=1))
    df["score_manual_std"] = df[score_cols].std(axis=1).fillna(0.0)
    df["n_raters"] = df[score_cols].notna().sum(axis=1)
    
    # Clean invalid or incomplete rows
    df = df.dropna(subset=["score_manual_avg"])
    df = df[df["student_answer"].str.strip() != ""].reset_index(drop=True)
    
    # Reorder columns
    df = df[[
        "topic", "question_id", "question_text", "reference_answer",
        "student_id", "student_answer",
        "score_m1", "score_m2", "score_m3",
        "score_manual_avg", "score_manual_std", "n_raters"
    ]]
    
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple:
    """Split the dataset into stratified train and test partitions.

    Bins continuous scores to perform stratified splitting, ensuring identical score distributions.

    Args:
        df: The consolidated master DataFrame.
        test_size: Proportion of dataset to include in test split.
        seed: Random state seed.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Create bin boundaries for stratification
    min_score = df["score_manual_avg"].min()
    max_score = df["score_manual_avg"].max()
    
    # Define 5 bins across the score range
    bins = np.linspace(min_score, max_score, 6)
    score_bins = np.digitize(df["score_manual_avg"], bins)
    
    # Stratify by a combination of topic and score bin
    strat_key = df["topic"].astype(str) + "_" + score_bins.astype(str)
    
    # Handle single-instance bins by merging them to ensure stratification is possible
    counts = strat_key.value_counts()
    singletons = counts[counts < 2].index
    if not singletons.empty:
        strat_key = strat_key.apply(lambda k: "other" if k in singletons else k)
        
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat_key
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing and train/test partition pipeline.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()
    
    print(f"[Preprocess] Initializing pipeline...")
    print(f"[Preprocess] Project Root: {PROJECT_ROOT}")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build master
    df = build_master_dataset()
    print(f"[Preprocess] Parsed {len(df)} student answers.")
    
    # Normalize score to [0, 1] range for regression heads (score range is 0 to 100)
    max_possible_score = 100.0
    df["score_norm"] = df["score_manual_avg"] / max_possible_score
    
    # Perform splits
    train_df, test_df = split_dataset(df, test_size=args.test_size, seed=args.seed)
    print(f"[Preprocess] Train set: {len(train_df)} rows.")
    print(f"[Preprocess] Test set:  {len(test_df)} rows.")
    
    # Save partitions as Parquet
    master_path = PROCESSED_DIR / "dataset_master.parquet"
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    
    df.to_parquet(master_path, index=False)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"[Preprocess] [OK] Saved master dataset to: {master_path}")
    print(f"[Preprocess] [OK] Saved stratified train partition to: {train_path}")
    print(f"[Preprocess] [OK] Saved stratified test partition to: {test_path}")


if __name__ == "__main__":
    main()
