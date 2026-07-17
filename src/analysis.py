"""Trains and evaluates three jailbreak-detection models (TF-IDF, linguistic,
combined), runs ablation studies and error analysis, and saves all results
to the results/ directory."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from features import FeatureExtractor

RESULTS_DIR = "results"
FEATURE_GROUPS = {
    "structure": ["avg_sentence_len", "avg_word_len", "unique_ratio",
                  "log_length", "caps_ratio", "special_ratio"],
    "attack_markers": ["persona_rate", "instruction_rate", "jailbreak_name_rate"],
    "rule_references": ["system_ref_rate", "hypothetical_rate"],
    "obfuscation": ["obfuscation_rate", "obfuscation_char_rate", "bracket_rate"],
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted")
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {"precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc), 4)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv("data/prompts.csv").dropna(subset=["prompt"]).reset_index(drop=True)
    fx = FeatureExtractor()
    features = pd.DataFrame([fx.extract(t) for t in df["prompt"]])

    y = df["label"].values
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=42, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_ling = scaler.fit_transform(features.values)

    tfidf = TfidfVectorizer(max_features=500, stop_words="english", min_df=2)
    X_tfidf = tfidf.fit_transform(df["prompt"])

    X_comb = hstack([X_tfidf, csr_matrix(X_ling)]).tocsr()

    results = {}

    m_tfidf = LogisticRegression(max_iter=2000, random_state=42)
    m_tfidf.fit(X_tfidf[train_idx], y_train)
    results["tfidf"] = evaluate(m_tfidf, X_tfidf[test_idx], y_test)

    m_ling = LogisticRegression(max_iter=2000, random_state=42)
    m_ling.fit(X_ling[train_idx], y_train)
    results["linguistic"] = evaluate(m_ling, X_ling[test_idx], y_test)

    m_comb = LogisticRegression(max_iter=2000, random_state=42)
    m_comb.fit(X_comb[train_idx], y_train)
    results["combined"] = evaluate(m_comb, X_comb[test_idx], y_test)

    with open(f"{RESULTS_DIR}/model_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    importance = pd.DataFrame({
        "feature": features.columns,
        "weight": m_ling.coef_[0],
    })
    importance["abs_weight"] = importance["weight"].abs()
    importance = importance.sort_values("abs_weight", ascending=False)
    importance[["feature", "weight"]].to_csv(
        f"{RESULTS_DIR}/feature_importance.csv", index=False)

    ablation = {"linguistic_only": {}, "combined": {}}
    base_ling = results["linguistic"]["f1"]
    base_comb = results["combined"]["f1"]
    for group, cols in FEATURE_GROUPS.items():
        keep = [c for c in features.columns if c not in cols]
        keep_i = [list(features.columns).index(c) for c in keep]
        X_ling_r = X_ling[:, keep_i]

        mr = LogisticRegression(max_iter=2000, random_state=42)
        mr.fit(X_ling_r[train_idx], y_train)
        f1_r = evaluate(mr, X_ling_r[test_idx], y_test)["f1"]
        ablation["linguistic_only"][group] = {
            "f1": f1_r, "drop": round(base_ling - f1_r, 4)}

        X_comb_r = hstack([X_tfidf, csr_matrix(X_ling_r)]).tocsr()
        mrc = LogisticRegression(max_iter=2000, random_state=42)
        mrc.fit(X_comb_r[train_idx], y_train)
        f1_rc = evaluate(mrc, X_comb_r[test_idx], y_test)["f1"]
        ablation["combined"][group] = {
            "f1": f1_rc, "drop": round(base_comb - f1_rc, 4)}

    with open(f"{RESULTS_DIR}/ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    probs = m_comb.predict_proba(X_comb[test_idx])[:, 1]
    preds = m_comb.predict(X_comb[test_idx])
    err = pd.DataFrame({
        "prompt": df["prompt"].values[test_idx],
        "true_label": y_test,
        "predicted": preds,
        "jailbreak_probability": probs.round(3),
    })
    false_pos = err[(err.true_label == 0) & (err.predicted == 1)].copy()
    false_neg = err[(err.true_label == 1) & (err.predicted == 0)].copy()
    false_pos["error_type"] = "false_positive"
    false_neg["error_type"] = "false_negative"
    errors = pd.concat([false_pos, false_neg])
    errors["prompt"] = errors["prompt"].str.slice(0, 300)
    errors.to_csv(f"{RESULTS_DIR}/errors.csv", index=False)

    _make_chart(results, f"{RESULTS_DIR}/model_comparison.png")
    _make_importance_chart(importance, f"{RESULTS_DIR}/feature_importance.png")

    print("Analysis complete. Results written to results/")
    print(f"  TF-IDF     F1={results['tfidf']['f1']}  AUC={results['tfidf']['auc']}")
    print(f"  Linguistic F1={results['linguistic']['f1']}  AUC={results['linguistic']['auc']}")
    print(f"  Combined   F1={results['combined']['f1']}  AUC={results['combined']['auc']}")
    print(f"  False positives: {len(false_pos)}, False negatives: {len(false_neg)}")


def _make_chart(results, path):
    models = ["tfidf", "linguistic", "combined"]
    labels = ["TF-IDF", "Linguistic", "Combined"]
    f1s = [results[m]["f1"] for m in models]
    aucs = [results[m]["auc"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w/2, f1s, w, label="F1", color="#3b6ea5")
    plt.bar(x + w/2, aucs, w, label="AUC", color="#a5c4e0")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    for i, v in enumerate(f1s):
        plt.text(i - w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(aucs):
        plt.text(i + w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _make_importance_chart(importance, path):
    top = importance.head(10).iloc[::-1]
    colors = ["#c0392b" if w > 0 else "#2980b9" for w in top["weight"]]
    plt.figure(figsize=(7, 4.5))
    plt.barh(top["feature"], top["weight"], color=colors)
    plt.xlabel("Logistic regression weight (positive = jailbreak)")
    plt.title("Feature Importance (linguistic model)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
