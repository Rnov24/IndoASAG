"""Handcrafted morphological and lexical feature extraction for Indonesian text.

Utilizes the Sastrawi stemmer to extract features categorized into 5 groups:
    - length:    ans_wc, ans_cc, len_ratio
    - overlap:   kw_overlap_n, kw_overlap_r, kw_coverage
    - sastrawi:  root_overlap
    - diversity: jaccard, ttr
    - structure: bigram_ov, completeness
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

# Lazy-load Sastrawi stemmer (expensive initialization)
_stemmer = None
_sastrawi_available = None


def _get_stemmer():
    """Lazy-initialize Sastrawi stemmer only when root_overlap is called."""
    global _stemmer, _sastrawi_available
    if _sastrawi_available is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
            _sastrawi_available = True
        except ImportError:
            _sastrawi_available = False
            print("[WARNING] Sastrawi library not found. 'root_overlap' will default to 0.0.")
    return _stemmer


# Feature groups for modular ablation tests
FEATURE_GROUPS = {
    "length":    ["ans_wc", "ans_cc", "len_ratio"],
    "overlap":   ["kw_overlap_n", "kw_overlap_r", "kw_coverage"],
    "sastrawi":  ["root_overlap"],
    "diversity": ["jaccard", "ttr"],
    "structure": ["bigram_ov", "completeness"],
}

ALL_FEATURES = []
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


def get_feature_names(groups: Optional[List[str]] = None) -> List[str]:
    """Retrieve named feature columns for active groups.

    Args:
        groups: List of group names (e.g., ["length", "sastrawi"]).
                If None, defaults to all 11 features.

    Returns:
        List of feature column names.
    """
    if groups is None:
        return ALL_FEATURES.copy()
        
    names = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group '{g}'. Available: {list(FEATURE_GROUPS.keys())}")
        names.extend(FEATURE_GROUPS[g])
    return names


def extract_row_features(ans: str, ref: str) -> Dict[str, float]:
    """Compute the 11 lexical/morphological features for a single answer-reference pair.

    Args:
        ans: The student's short answer string.
        ref: The instructor's reference answer string.

    Returns:
        A dictionary mapping feature names to their float values.
    """
    ans_clean = str(ans).lower()
    ref_clean = str(ref).lower()
    
    # Tokenize by whitespace
    ans_words = ans_clean.split()
    ref_words = ref_clean.split()
    
    ans_set = set(ans_words)
    ref_set = set(ref_words)
    
    overlap = ans_set & ref_set
    
    feats = {}
    
    # 1. Length features
    feats["ans_wc"] = float(len(ans_words))
    feats["ans_cc"] = float(len(ans_clean))
    feats["len_ratio"] = len(ans_words) / max(len(ref_words), 1.0)
    
    # 2. Overlap features
    feats["kw_overlap_n"] = float(len(overlap))
    feats["kw_overlap_r"] = len(overlap) / max(len(ref_set), 1.0)
    feats["kw_coverage"] = len(overlap) / max(len(ans_set), 1.0)
    
    # 3. Sastrawi morphological root overlap
    stemmer = _get_stemmer()
    if stemmer is not None:
        ans_stemmed = stemmer.stem(ans_clean).split()
        ref_stemmed = stemmer.stem(ref_clean).split()
        ans_stems = set(ans_stemmed)
        ref_stems = set(ref_stemmed)
        feats["root_overlap"] = len(ans_stems & ref_stems) / max(len(ref_stems), 1.0)
    else:
        feats["root_overlap"] = 0.0
        
    # 4. Diversity features
    union = ans_set | ref_set
    feats["jaccard"] = len(overlap) / max(len(union), 1.0)
    feats["ttr"] = len(ans_set) / max(len(ans_words), 1.0)
    
    # 5. Structure features
    # Bigram overlap
    ans_bigrams = set(zip(ans_words[:-1], ans_words[1:])) if len(ans_words) > 1 else set()
    ref_bigrams = set(zip(ref_words[:-1], ref_words[1:])) if len(ref_words) > 1 else set()
    feats["bigram_ov"] = len(ans_bigrams & ref_bigrams) / max(len(ref_bigrams), 1.0)
    
    # Clause completeness (checks reference segments divided by commas)
    ref_clauses = ref_clean.split(",")
    matched_clauses = 0
    for clause in ref_clauses:
        clause_words = [w for w in clause.split() if len(w) > 3]
        if not clause_words:
            # If the clause is empty or contains only short words, count it as matched to prevent bias
            matched_clauses += 1
            continue
        # Check if any word in the reference clause is present in the student's answer
        if any(w in ans_clean for w in clause_words):
            matched_clauses += 1
            
    feats["completeness"] = matched_clauses / max(len(ref_clauses), 1.0)
    
    return feats


def extract_features_df(df: pd.DataFrame,
                        groups: Optional[List[str]] = None,
                        ans_col: str = "student_answer",
                        ref_col: str = "reference_answer") -> pd.DataFrame:
    """Extract morphological features from a pandas DataFrame.

    Args:
        df: DataFrame containing answer and reference columns.
        groups: Selected feature groups to extract. If None, extracts all.
        ans_col: Name of student answer column.
        ref_col: Name of reference answer column.

    Returns:
        A DataFrame containing only the requested feature columns.
    """
    target_features = get_feature_names(groups)
    print(f"[Features] Extracting {len(target_features)} morphological features for {len(df)} rows...")
    
    rows = []
    for idx, row in df.iterrows():
        rows.append(extract_row_features(row[ans_col], row[ref_col]))
        
    feats_df = pd.DataFrame(rows, index=df.index)
    return feats_df[target_features]
