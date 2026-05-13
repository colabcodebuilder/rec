"""
Synthetic data generation for patients and providers.
"""
import pandas as pd
import numpy as np
from typing import Set, List, Dict, Tuple
import config


def generate_patient_ids(n: int = config.N_PATIENTS) -> List[str]:
    """
    Generate sequential patient IDs.
    """
    return [
        f"{config.PATIENT_PREFIX}{i:0{config.PATIENT_ZFILL}d}"
        for i in range(1, n + 1)
    ]


def generate_provider_ids(n: int = config.N_PROVIDERS) -> List[str]:
    """
    Generate sequential provider (doctor) IDs.
    """
    return [
        f"{config.PROVIDER_PREFIX}{i:0{config.PROVIDER_ZFILL}d}"
        for i in range(1, n + 1)
    ]


def generate_patients_df() -> pd.DataFrame:
    """
    Generate synthetic patient DataFrame with demographics and conditions.
    """
    patients_list = generate_patient_ids()
    n = len(patients_list)

    df = pd.DataFrame({
        "patient_id": patients_list,
        "age": np.random.randint(config.AGE_MIN, config.AGE_MAX + 1, size=n),
        "sex": np.random.choice(["M", "F"], size=n),
        "condition_tags_set": [
            set(np.random.choice(
                config.CONDITIONS,
                np.random.randint(1, 4),
                replace=False
            ))
            for _ in range(n)
        ],
        "risk_score": np.random.uniform(*config.RISK_SCORE_RANGE, size=n),
        "urgency_score": np.random.uniform(*config.URGENCY_SCORE_RANGE, size=n)
    })

    df["condition_tags"] = df["condition_tags_set"].apply(lambda s: ",".join(sorted(s)))
    return df


def generate_providers_df() -> pd.DataFrame:
    """
    Generate synthetic provider DataFrame with specialties and metrics.
    """
    providers_list = generate_provider_ids()
    n = len(providers_list)
    specialties = list(config.SPECIALITY_MAP.keys())

    df = pd.DataFrame({
        "provider_id": providers_list,
        "speciality": np.random.choice(specialties, size=n),
        "expertise_score": np.random.uniform(*config.EXPERTISE_SCORE_RANGE, size=n),
        "success_rate": np.random.uniform(*config.SUCCESS_RATE_RANGE, size=n),
        "availability_score": np.random.uniform(*config.AVAILABILITY_SCORE_RANGE, size=n),
        "load_score": np.random.uniform(*config.LOAD_SCORE_RANGE, size=n)
    })

    df["conditions_treated_set"] = df["speciality"].apply(
        lambda s: set(config.SPECIALITY_MAP[s])
    )
    df["conditions_treated"] = df["conditions_treated_set"].apply(
        lambda s: ",".join(sorted(s))
    )
    return df


def tag_string_to_set(tag_string: str) -> Set[str]:
    """
    Convert comma-separated tag string to a set of lowercase strings.
    """
    return set(
        t.strip().lower()
        for t in str(tag_string).split(",")
        if t.strip()
    )


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_overlap_score(set_a: Set[str], set_b: Set[str]) -> int:
    """
    Count number of overlapping elements between two sets.
    """
    return len(set_a & set_b)