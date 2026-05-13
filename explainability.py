"""
SHAP-based explainability for recommendation decisions.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

import config
from recommendation_engine import build_features

logger = logging.getLogger(__name__)


class RecommendationExplainer:
    """
    SHAP-based explainer for healthcare provider recommendations.
    """

    def __init__(self, model: Optional[GradientBoostingRegressor] = None):
        """
        Initialize explainer with optional pre-trained model.

        Args:
            model: Pre-trained GradientBoostingRegressor or None to use dummy
        """
        self.feature_names = config.FEATURE_NAMES
        self.model = model if model is not None else self._build_surrogate_model()
        self.explainer = shap.Explainer(self.model, feature_names=self.feature_names)

    @staticmethod
    def _build_surrogate_model() -> GradientBoostingRegressor:
        """
        Build a dummy GradientBoosting model for SHAP explainer initialization.
        """
        X_dummy = np.random.rand(100, len(config.FEATURE_NAMES))
        weights = np.array([0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
        y_dummy = X_dummy @ weights + np.random.randn(100) * 0.01
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=config.RANDOM_SEED
        )
        model.fit(X_dummy, y_dummy)
        return model

    def explain(
        self,
        patient: Dict[str, Any],
        provider: Dict[str, Any],
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Generate SHAP explanation for a patient-provider pair.
        """
        features_list = build_features(patient, provider)
        features_array = np.array(features_list, dtype=np.float32).reshape(1, -1)

        logger.info(
            "Explaining Patient %s → Provider %s",
            patient.get("patient_id", "N/A"),
            provider.get("provider_id", "N/A")
        )

        shap_values_obj = self.explainer(features_array)

        explanation_df = pd.DataFrame({
            "feature": self.feature_names,
            "value": features_array.flatten(),
            "shap_value": shap_values_obj.values[0]
        })

        explanation_df["impact"] = np.where(
            explanation_df["shap_value"] > 0,
            "↑ increases score",
            "↓ decreases score"
        )
        explanation_df = explanation_df.sort_values("shap_value", ascending=False)

        if debug:
            logger.info("Base value: %.4f", shap_values_obj.base_values[0])

        return explanation_df

    def plot_summary(self, explanation_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Visualize SHAP values as a horizontal bar chart.
        """
        plt.figure(figsize=(10, 5))
        colors = [
            "#2ecc71" if x > 0 else "#e74c3c"
            for x in explanation_df["shap_value"]
        ]

        sns.barplot(
            data=explanation_df,
            x="shap_value",
            y="feature",
            palette=colors
        )
        plt.axvline(0, color='black', lw=1)
        plt.title("Feature Impact on Recommendation Score")
        plt.xlabel("SHAP Value (Contribution)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP plot saved to {save_path}")
        else:
            plt.show()

    def explain_batch(
        self,
        patient: Dict[str, Any],
        providers: list,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate explanations for top-K providers for a patient.

        Args:
            patient: Patient data dictionary
            providers: List of provider data dictionaries
            top_k: Number of top recommendations to explain

        Returns:
            Dictionary with explanations for top providers
        """
        from recommendation_engine import fallback_score

        # Score all providers
        scored = []
        for provider in providers:
            features = build_features(patient, provider)
            score = fallback_score(features)
            scored.append((provider, features, score))

        # Sort and take top K
        scored.sort(key=lambda x: x[2], reverse=True)
        top_providers = scored[:top_k]

        # Generate explanations
        results = {
            "patient_id": patient.get("patient_id"),
            "explanations": []
        }

        for provider, features, score in top_providers:
            explanation = self.explain(patient, provider)
            results["explanations"].append({
                "provider_id": provider.get("provider_id"),
                "speciality": provider.get("speciality"),
                "score": round(score, 4),
                "shap_values": explanation.to_dict("records")
            })

        return results


def build_explanation_summary(
    features: list,
    common_conditions: list,
    provider_speciality: str
) -> Dict[str, Any]:
    """
    Build a human-readable explanation summary for API responses.
    """
    w = config.SCORE_WEIGHTS
    return {
        "matched_conditions": list(common_conditions),
        "score_breakdown": {
            "medical_fit_impact": round(w["jaccard"] * features[0], 3),
            "expertise_impact": round(w["expertise"] * features[3], 3),
            "load_balancing_factor": round(features[6], 2)
        },
        "reason": (
            f"Specializes in {provider_speciality} and treats "
            f"your condition(s): {', '.join(common_conditions)}"
        ) if common_conditions else (
            "General match based on risk scores and availability."
        )
    }