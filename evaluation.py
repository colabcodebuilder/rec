"""
Evaluation metrics and analysis for the recommendation system.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from typing import Dict, List, Tuple, Any, Optional

import config
from recommendation_engine import RecommendationEngine
from data_generator import compute_jaccard_similarity, tag_string_to_set

logger = logging.getLogger(__name__)


def assign_ground_truth_label(
    jaccard: float,
    patient: Dict[str, Any],
    provider: Dict[str, Any]
) -> int:
    """
    Assign ground truth relevance label (0-3) based on weighted criteria.
    """
    score = (
        jaccard * 5 +
        float(patient.get("risk_score", 1.0)) * 1.5 +
        float(provider.get("expertise_score", 1.0)) * 2 +
        float(provider.get("success_rate", 0.8)) * 2
    )

    thresholds = config.GROUND_TRUTH_THRESHOLDS
    if score >= thresholds["excellent"]:
        return 3
    if score >= thresholds["good"]:
        return 2
    if score >= thresholds["fair"]:
        return 1
    return 0


def compute_ndcg(rel_scores: List[int], k: int = config.NDCG_K) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.
    """
    def dcg(relevances: List[int]) -> float:
        return np.sum([
            (2 ** r - 1) / np.log2(i + 2)
            for i, r in enumerate(relevances[:k])
        ])

    actual_dcg = dcg(rel_scores)
    ideal_dcg = dcg(sorted(rel_scores, reverse=True))
    return actual_dcg / (ideal_dcg + 1e-6)


def evaluate_recommendations(engine: RecommendationEngine) -> Dict[str, Any]:
    """
    Evaluate recommendation quality using NDCG@K and provider hit distribution.
    """
    ndcg_scores: List[float] = []
    provider_hits: Dict[str, int] = {}

    providers_records = engine.providers_df.to_dict("records")

    for _, patient in engine.patients_df.iterrows():
        labels = []
        scores = []

        # Build predictions (from KNN or random baseline)
        if patient["patient_id"] in engine.matrix.index:
            p_idx = engine.matrix.index.get_loc(patient["patient_id"])
            _, indices = engine.knn_model.kneighbors(
                engine.matrix.iloc[p_idx, :].values.reshape(1, -1),
                n_neighbors=min(config.KNN_NEIGHBORS, len(engine.matrix))
            )
            pred_scores = engine.matrix.iloc[indices.flatten()[1:]].mean(axis=0)
        else:
            pred_scores = pd.Series(0, index=engine.matrix.columns)

        p_tags = patient["condition_tags_set"]
        for provider in providers_records:
            d_tags = provider["conditions_treated_set"]
            jaccard = compute_jaccard_similarity(p_tags, d_tags)

            labels.append(assign_ground_truth_label(jaccard, patient, provider))
            base_score = pred_scores.get(provider["provider_id"], 0)
            scores.append(base_score + np.random.normal(0, 0.05))  # noise for tie-breaking

        # Rank by score descending
        ranked_idx = np.argsort(scores)[::-1]
        ndcg_scores.append(compute_ndcg([labels[i] for i in ranked_idx]))

        # Track top provider
        top_provider = providers_records[ranked_idx[0]]["provider_id"]
        provider_hits[top_provider] = provider_hits.get(top_provider, 0) + 1

    results = {
        "mean_ndcg": np.mean(ndcg_scores),
        "ndcg_scores": ndcg_scores,
        "provider_hits": provider_hits,
        "ndcg_df": pd.DataFrame({"ndcg@5": ndcg_scores}),
        "top_providers_df": pd.DataFrame(
            provider_hits.items(),
            columns=["provider_id", "count"]
        ).sort_values("count", ascending=False)
    }

    logger.info(f"Mean NDCG@{config.NDCG_K}: {results['mean_ndcg']:.4f}")
    return results


def plot_evaluation(eval_results: Dict[str, Any]) -> None:
    """
    Generate Plotly visualizations for evaluation metrics.
    """
    # NDCG distribution
    fig1 = px.histogram(
        eval_results["ndcg_df"],
        x="ndcg@5",
        title="NDCG@5 Distribution"
    )
    fig1.show()

    # Provider recommendation bias
    fig2 = px.bar(
        eval_results["top_providers_df"].head(10),
        x="provider_id",
        y="count",
        title="Recommendation Bias (Top 10 Providers)"
    )
    fig2.show()