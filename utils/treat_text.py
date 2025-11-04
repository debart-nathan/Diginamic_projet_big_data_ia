import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict

def get_pos_and_neg_words(df_merged):
    df_en = df_merged[
        (df_merged["FeedbackText"].notna()) & 
        (df_merged["SatisfactionScore"].notna())
    ].copy()

    # Tokenize and collect scores
    word_scores = defaultdict(list)
    for text, score in zip(df_en["FeedbackText"], df_en["SatisfactionScore"]):
        words = re.findall(r'\b[a-zA-Z]{2,}\b', str(text).lower())
        filtered_words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        for word in filtered_words:
            word_scores[word].append(score)

    # Compute average score and count per word
    word_stats = {
        word: {"avg_score": np.mean(scores), "count": len(scores)}
        for word, scores in word_scores.items() if len(scores) >= 2
    }

    df_scores = pd.DataFrame([
        {"token": word, "avg_score": stats["avg_score"], "count": stats["count"]}
        for word, stats in word_stats.items()
    ])

    # Thresholds
    pos_thresh = df_scores["avg_score"].quantile(0.90)
    neg_thresh = df_scores["avg_score"].quantile(0.10)

    # Select positive and negative words with stats
    positive_words = df_scores[df_scores["avg_score"] >= pos_thresh][["token", "avg_score", "count"]]
    negative_words = df_scores[df_scores["avg_score"] <= neg_thresh][["token", "avg_score", "count"]]

    return positive_words, negative_words
