"""
Core recommendation logic using KNN collaborative filtering.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import logging
import warnings

import config
from data_generator import (
    generate_patients_df,
    generate_providers_df,
    compute_jaccard_similarity,
    tag_string_to_set
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Healthcare provider recommendation engine using KNN collaborative filtering.
    """

    def __init__(self):
        self.patients_df: Optional[pd.DataFrame] = None
        self.providers_df: Optional[pd.DataFrame] = None
        self.matrix: Optional[pd.DataFrame] = None
        self.knn_model: Optional[NearestNeighbors] = None
        self.provider_info_map: Dict[str, Dict] = {}
        self.provider_treats_map: Dict[str, Set[str]] = {}
        self._initialize()

    def _initialize(self) -> None:
        """
        Generate data, build interaction matrix, and fit KNN model.
        """
        logger.info("Generating synthetic patient/provider data...")
        self.patients_df = generate_patients_df()
        self.providers_df = generate_providers_df()

        logger.info("Building interaction matrix...")
        self._build_interaction_matrix()

        logger.info("Fitting KNN model...")
        self._fit_knn()

        logger.info("Building provider lookup maps...")
        self._build_provider_maps()

    def _build_interaction_matrix(self) -> None:
        """
        Create patient × provider interaction matrix based on condition overlap.
        """
        rows = []
        for _, patient in self.patients_df.iterrows():
            p_tags = patient["condition_tags_set"]
            for _, provider in self.providers_df.iterrows():
                d_tags = provider["conditions_treated_set"]
                overlap = len(p_tags & d_tags)
                if overlap > 0:
                    rows.append({
                        "patient_id": patient["patient_id"],
                        "provider_id": provider["provider_id"],
                        "score": overlap
                    })

        df_inter = pd.DataFrame(rows)
        if df_inter.empty:
            raise ValueError("No interaction data generated. Check condition mapping.")

        self.matrix = df_inter.pivot(
            index="patient_id",
            columns="provider_id",
            values="score"
        ).fillna(0)

    def _fit_knn(self) -> None:
        """
        Fit NearestNeighbors model on the interaction matrix.
        """
        self.knn_model = NearestNeighbors(
            metric=config.KNN_METRIC,
            algorithm=config.KNN_ALGORITHM,
            n_neighbors=min(config.KNN_NEIGHBORS, len(self.matrix))
        )
        self.knn_model.fit(csr_matrix(self.matrix.values))

    def _build_provider_maps(self) -> None:
        """
        Build fast lookup dictionaries for provider info and treatable conditions.
        """
        self.provider_info_map = self.providers_df.set_index("provider_id").to_dict("index")
        self.provider_treats_map = {
            pid: row["conditions_treated_set"]
            for pid, row in self.providers_df.set_index("provider_id").iterrows()
        }

    def get_patient_conditions(self, patient_id: str) -> Optional[Set[str]]:
        """
        Get the condition set for a patient.
        """
        match = self.patients_df[self.patients_df["patient_id"] == patient_id]
        if match.empty:
            return None
        return match.iloc[0]["condition_tags_set"]

    def get_recommendations(
        self,
        patient_id: str,
        n_recs: int = config.TOP_K_RECS
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate top-N provider recommendations for a given patient.
        Scores blend collaborative filtering with provider quality metrics.
        """
        if patient_id not in self.matrix.index:
            return None

        # Get patient conditions and full patient data dict
        p_conds = self.get_patient_conditions(patient_id)
        patient_row = self.patients_df[self.patients_df["patient_id"] == patient_id].iloc[0]
        patient_dict = patient_row.to_dict()

        # Only providers that can treat at least one patient condition
        qualified_docs = [
            doc for doc, treats in self.provider_treats_map.items()
            if p_conds & treats
        ]
        if not qualified_docs:
            return None

        # Collaborative scores from KNN
        idx = self.matrix.index.get_loc(patient_id)
        _, indices = self.knn_model.kneighbors(
            self.matrix.iloc[idx, :].values.reshape(1, -1),
            n_neighbors=min(config.KNN_NEIGHBORS, len(self.matrix))
        )
        similar_indices = indices.flatten()[1:]  # exclude self
        collab_scores = self.matrix.iloc[similar_indices].mean(axis=0)

        # Build results with hybrid scoring
        results = []
        for doc_id in qualified_docs:
            info = self.provider_info_map[doc_id]

            # Compute quality score using feature-based weighting
            features = build_features(patient_dict, info)
            quality_score = fallback_score(features)   # range ~0 to ~5

            # Normalize collaborative score (0-1)
            collab = collab_scores.get(doc_id, 0)
            max_collab = collab_scores.max() if not collab_scores.empty else 1e-6
            collab_norm = min(collab / max_collab, 1.0)

            # Blend: 60% quality, 40% collaborative, tiny noise to break ties
            final_score = 0.6 * quality_score + 0.4 * collab_norm + np.random.uniform(0, 0.0001)

            common_conds = p_conds & self.provider_treats_map[doc_id]
            results.append({
                "provider_id": doc_id,
                "speciality": info["speciality"],
                "expertise_score": info["expertise_score"],
                "success_rate": info["success_rate"],
                "score": round(float(final_score), 4),
                "matched_conditions": sorted(common_conds),
                "reason": f"Specializes in {info['speciality']} and treats: {', '.join(sorted(common_conds))}"
            })

        # Sort descending and return top N
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_recs]


def build_features(patient: Dict[str, Any], provider: Dict[str, Any]) -> List[float]:
    """
    Build feature vector for a patient-provider pair.
    """
    p_tags = tag_string_to_set(patient.get("condition_tags", ""))
    d_tags = tag_string_to_set(provider.get("conditions_treated", ""))

    jaccard = compute_jaccard_similarity(p_tags, d_tags)
    inverse_load = 1 - float(provider.get("load_score", 0.5))

    return [
        jaccard,
        float(patient.get("risk_score", 1.0)),
        float(patient.get("urgency_score", 1.0)),
        float(provider.get("expertise_score", 1.0)),
        float(provider.get("success_rate", 0.8)),
        float(provider.get("availability_score", 0.8)),
        inverse_load
    ]


def fallback_score(features: List[float]) -> float:
    """
    Compute recommendation score using weighted linear combination.
    """
    w = config.SCORE_WEIGHTS
    weighted_sum = (
        w["jaccard"] * features[0] +
        w["risk"] * features[1] +
        w["urgency"] * features[2] +
        w["expertise"] * features[3] +
        w["success"] * features[4] +
        w["availability"] * features[5]
    )
    # Multiply by inverse load factor
    return weighted_sum * features[6]