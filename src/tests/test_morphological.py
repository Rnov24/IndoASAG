"""Unit tests for morphological feature extraction.

Run using: pytest src/tests/
"""

import pytest
import pandas as pd
from src.morphological import extract_row_features, extract_features_df, get_feature_names


def test_feature_names():
    """Verify feature names retrieval."""
    all_names = get_feature_names()
    assert len(all_names) == 11
    assert "ans_wc" in all_names
    assert "root_overlap" in all_names
    
    length_names = get_feature_names(["length"])
    assert len(length_names) == 3
    assert "ans_wc" in length_names
    assert "root_overlap" not in length_names


def test_extract_row_identical():
    """Verify features when student and reference answers are identical."""
    ans = "Saya sedang makan nasi goreng"
    ref = "Saya sedang makan nasi goreng"
    
    feats = extract_row_features(ans, ref)
    
    assert feats["ans_wc"] == 5.0
    assert feats["ans_cc"] == len(ans.lower())
    assert feats["len_ratio"] == 1.0
    assert feats["kw_overlap_n"] == 5.0
    assert feats["kw_overlap_r"] == 1.0
    assert feats["kw_coverage"] == 1.0
    assert feats["jaccard"] == 1.0
    assert feats["bigram_ov"] == 1.0
    assert feats["completeness"] == 1.0
    
    # Sastrawi should stem and overlap should be 1.0 if Sastrawi is installed
    assert feats["root_overlap"] == 1.0


def test_extract_row_completely_different():
    """Verify features when student and reference answers share zero words."""
    ans = "Kucing itu tidur di atas kasur"
    ref = "Dunia teknologi berkembang sangat pesat"
    
    feats = extract_row_features(ans, ref)
    
    assert feats["kw_overlap_n"] == 0.0
    assert feats["kw_overlap_r"] == 0.0
    assert feats["kw_coverage"] == 0.0
    assert feats["jaccard"] == 0.0
    assert feats["bigram_ov"] == 0.0
    assert feats["root_overlap"] == 0.0
    assert feats["completeness"] == 0.0


def test_extract_row_partial():
    """Verify features with partial overlap and Indonesian stemming."""
    # "memakan" stems to "makan", so it should match "makan" in root_overlap,
    # but not in raw word overlap.
    ans = "Siswa memakan roti"
    ref = "Siswa makan roti tawar"
    
    feats = extract_row_features(ans, ref)
    
    # Raw word overlap: "siswa", "roti" (2 words overlap)
    # Reference unique words: "siswa", "makan", "roti", "tawar" (4 words)
    # Student unique words: "siswa", "memakan", "roti" (3 words)
    assert feats["kw_overlap_n"] == 2.0
    assert feats["kw_overlap_r"] == 2.0 / 4.0  # 0.5
    assert feats["kw_coverage"] == 2.0 / 3.0   # 0.666...
    
    # Stemmed overlap:
    # Student stems: "siswa", "makan", "roti" (3 stems)
    # Reference stems: "siswa", "makan", "roti", "tawar" (4 stems)
    # Overlap stems: "siswa", "makan", "roti" (3 stems overlap)
    # Root overlap ratio = 3 / 4 = 0.75
    assert feats["root_overlap"] == 0.75


def test_extract_features_df():
    """Verify DataFrame feature extraction."""
    data = {
        "student_answer": ["makan nasi", "tidur saja"],
        "reference_answer": ["makan nasi goreng", "tidur di kasur"]
    }
    df = pd.DataFrame(data)
    
    feats_df = extract_features_df(df, groups=["length", "overlap"])
    
    assert len(feats_df) == 2
    assert list(feats_df.columns) == ["ans_wc", "ans_cc", "len_ratio", "kw_overlap_n", "kw_overlap_r", "kw_coverage"]
    assert feats_df.loc[0, "ans_wc"] == 2.0
    assert feats_df.loc[0, "kw_overlap_n"] == 2.0
