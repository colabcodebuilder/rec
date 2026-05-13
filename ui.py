"""
Streamlit web application for Healthcare Recommendation System.
"""
import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from database import get_connection, fetch_patient, fetch_all_providers, seed_database
from recommendation_engine import (
    RecommendationEngine,
    build_features,
    fallback_score
)
from explainability import RecommendationExplainer
from mlflow_tracking import MLflowManager
from onnx_export import ONNXPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .explanation-box {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Initialize Session State
# ==========================================
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'mlflow_mgr' not in st.session_state:
    st.session_state.mlflow_mgr = MLflowManager()
if 'onnx_predictor' not in st.session_state:
    st.session_state.onnx_predictor = None
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'onnx_loaded' not in st.session_state:
    st.session_state.onnx_loaded = False

# ==========================================
# Auto-load ONNX Model on Startup
# ==========================================
def auto_load_onnx():
    """Automatically load ONNX model if available."""
    if st.session_state.onnx_loaded and st.session_state.onnx_predictor is not None:
        return True
    
    onnx_dir = "onnx_models"
    if not os.path.exists(onnx_dir):
        logger.warning(f"ONNX directory not found: {onnx_dir}")
        return False
    
    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith('.onnx')]
    
    if not onnx_files:
        logger.warning("No ONNX files found")
        return False
    
    # Sort by size (largest first - usually the full model)
    onnx_files.sort(key=lambda f: os.path.getsize(os.path.join(onnx_dir, f)), reverse=True)
    
    for onnx_file in onnx_files:
        onnx_path = os.path.join(onnx_dir, onnx_file)
        try:
            logger.info(f"Attempting to load: {onnx_path}")
            predictor = ONNXPredictor(onnx_path)
            st.session_state.onnx_predictor = predictor
            st.session_state.onnx_loaded = True
            logger.info(f"✅ ONNX model auto-loaded: {onnx_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load {onnx_file}: {e}")
            continue
    
    return False

# Try auto-load on startup
if not st.session_state.onnx_loaded:
    auto_load_onnx()

# ==========================================
# Helper Functions
# ==========================================
@st.cache_resource
def load_recommendation_engine():
    """Load and cache the recommendation engine."""
    with st.spinner("Loading recommendation engine..."):
        engine = RecommendationEngine()
    return engine


def load_onnx_model(onnx_path: str) -> Optional[ONNXPredictor]:
    """Load ONNX model for fast inference."""
    try:
        if not os.path.exists(onnx_path):
            st.error(f"Model file not found: {onnx_path}")
            return None
        
        predictor = ONNXPredictor(onnx_path)
        st.session_state.onnx_loaded = True
        return predictor
    except Exception as e:
        st.error(f"Could not load ONNX model: {str(e)[:200]}")
        return None


def get_recommendations_from_api(
    patient_id: str,
    explain: bool = True,
    api_url: str = "http://localhost:8000"
) -> Optional[Dict[str, Any]]:
    """Get recommendations from the FastAPI server."""
    try:
        params = {"explain": explain}
        response = requests.get(
            f"{api_url}/recommend/{patient_id}",
            params=params,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        return None


def plot_score_breakdown(recommendation: Dict[str, Any]):
    """Create a score breakdown visualization."""
    if 'explanation' not in recommendation:
        return None

    breakdown = recommendation['explanation'].get('score_breakdown', {})
    if not breakdown:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            marker_color=['#4caf50', '#2196f3', '#ff9800'],
            text=[f"{v:.3f}" for v in breakdown.values()],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Score Breakdown",
        xaxis_title="Factor",
        yaxis_title="Impact",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_shap_waterfall(shap_values: List[Dict[str, Any]]):
    """Create a waterfall plot for SHAP values."""
    if not shap_values:
        return None

    df = pd.DataFrame(shap_values)
    df = df.sort_values('shap_value', ascending=True)

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(df),
        y=df['feature'].tolist(),
        x=df['shap_value'].tolist(),
        text=[f"{v:.3f}" for v in df['shap_value']],
        connector={"mode": "spanning", "line": {"width": 1}}
    ))
    fig.update_layout(
        title="SHAP Feature Impact",
        xaxis_title="SHAP Value",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ==========================================
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
    st.markdown("## 🏥 Navigation")

    page = st.radio(
        "Select Page",
        [
            "🏠 Home",
            "🔍 Recommendations",
            "📊 Model Analysis",
            "🔬 Explainability",
            "📈 MLflow Dashboard",
            "⚙️ Settings"
        ]
    )

    st.markdown("---")
    st.markdown("### System Status")

    # Check API status
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("API: Online ✅")
        else:
            st.warning("API: Degraded ⚠️")
    except:
        st.error("API: Offline ❌")

    # Check ONNX model
    if st.session_state.onnx_predictor:
        model_name = os.path.basename(st.session_state.onnx_predictor.onnx_path)
        st.success(f"ONNX Model: Loaded ✅")
        st.caption(f"📦 {model_name}")
    else:
        st.info("ONNX Model: Not loaded")
        if st.button("🔄 Try Auto-Load"):
            if auto_load_onnx():
                st.rerun()

    st.markdown("---")
    st.markdown(f"**Version:** 1.0.0")
    st.markdown(f"**Patients:** {config.N_PATIENTS}")
    st.markdown(f"**Providers:** {config.N_PROVIDERS}")


# ==========================================
# Page: Home
# ==========================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Healthcare Recommendation System</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Patients", config.N_PATIENTS)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Providers", config.N_PROVIDERS)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Medical Conditions", len(config.CONDITIONS))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### Welcome to the Healthcare Recommendation Engine

    This system provides **AI-powered provider recommendations** for patients based on:

    - 🏥 **Medical condition matching** (Jaccard similarity)
    - 📊 **Provider expertise and success rates**
    - ⚖️ **Workload balancing** (availability optimization)
    - 🔬 **Explainable AI** (SHAP feature importance)

    ### Features
    - **Smart Recommendations**: KNN collaborative filtering with medical logic
    - **Explainability**: SHAP values show why each provider is recommended
    - **MLflow Integration**: Track experiments, models, and metrics
    - **ONNX Export**: Optimized models for production deployment
    - **REST API**: FastAPI backend for integration

    ### Quick Start
    1. Go to **Recommendations** to find providers for a patient
    2. Check **Model Analysis** for performance metrics
    3. Explore **Explainability** to understand recommendations
    4. View **MLflow Dashboard** for experiment tracking
    """)

    # Show available conditions
    st.markdown("### Available Medical Conditions")
    cols = st.columns(4)
    for i, condition in enumerate(config.CONDITIONS):
        with cols[i % 4]:
            st.markdown(f"- {condition}")

    # Show specialties
    st.markdown("### Provider Specialties")
    specialties_df = pd.DataFrame([
        {"Specialty": k, "Treats": ", ".join(v)}
        for k, v in config.SPECIALITY_MAP.items()
    ])
    st.dataframe(specialties_df, use_container_width=True)


# ==========================================
# Page: Recommendations
# ==========================================
elif page == "🔍 Recommendations":
    st.markdown('<h1 class="main-header">🔍 Provider Recommendations</h1>', unsafe_allow_html=True)

    # Load engine if not loaded
    if st.session_state.engine is None:
        st.session_state.engine = load_recommendation_engine()

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        patient_options = [
            f"{config.PATIENT_PREFIX}{i:0{config.PATIENT_ZFILL}d}"
            for i in range(1, config.N_PATIENTS + 1)
        ]
        patient_id = st.selectbox(
            "Select Patient",
            options=patient_options,
            help="Choose a patient to get recommendations"
        )

    with col2:
        use_onnx = st.checkbox("Use ONNX Model", value=True)
        use_api = st.checkbox("Use API", value=False)
        n_recommendations = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            if use_api:
                results = get_recommendations_from_api(
                    patient_id,
                    explain=True,
                    api_url=st.session_state.api_url
                )
                if results:
                    recommendations = results.get("recommendations", [])
                    patient_conditions = results.get("patient_conditions", [])
                else:
                    st.error("API not available. Using local engine instead.")
                    results = st.session_state.engine.get_recommendations(patient_id, n_recommendations)
                    recommendations = results or []
                    patient_conditions = list(
                        st.session_state.engine.get_patient_conditions(patient_id) or []
                    )
            else:
                results = st.session_state.engine.get_recommendations(patient_id, n_recommendations)
                recommendations = results or []
                patient_conditions = list(
                    st.session_state.engine.get_patient_conditions(patient_id) or []
                )

            st.markdown("---")
            st.markdown(f"### Patient: {patient_id}")
            st.markdown(f"**Conditions:** {', '.join(sorted(patient_conditions))}")

            if recommendations:
                st.markdown(f"### Top {len(recommendations)} Providers")

                for i, rec in enumerate(recommendations):
                    with st.expander(
                        f"#{i+1} {rec['provider_id']} - {rec.get('speciality', 'General')} "
                        f"(Score: {rec['score']:.4f})",
                        expanded=(i == 0)
                    ):
                        col1, col2, col3 = st.columns([1, 1, 1])

                        with col1:
                            st.metric("Score", f"{rec['score']:.4f}")
                            st.metric("Specialty", rec.get('speciality', 'N/A'))

                        with col2:
                            matched = rec.get('matched_conditions', [])
                            st.markdown(f"**Matched Conditions:** {', '.join(matched)}")

                        with col3:
                            if 'explanation' in rec:
                                reason = rec['explanation'].get('reason', '')
                                st.info(reason)

                        if 'explanation' in rec:
                            fig = plot_score_breakdown(rec)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No matching providers found for this patient.")

    st.markdown("---")
    st.markdown("### Quick Test Patients")
    sample_patients = ["Patient_001", "Patient_010", "Patient_050", "Patient_099"]
    cols = st.columns(len(sample_patients))
    for i, pid in enumerate(sample_patients):
        with cols[i]:
            if st.button(f"Test {pid}", key=f"quick_{pid}"):
                st.session_state['quick_patient'] = pid
                st.rerun()


# ==========================================
# Page: Model Analysis
# ==========================================
elif page == "📊 Model Analysis":
    st.markdown('<h1 class="main-header">📊 Model Analysis</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Performance", "ONNX Models", "Feature Analysis"])

    with tab1:
        st.markdown("### MLflow Experiment Analysis")
        mlflow_mgr = st.session_state.mlflow_mgr
        runs_df = mlflow_mgr.search_runs(max_results=20)

        if not runs_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                if "metrics.test_rmse" in runs_df.columns:
                    fig = px.histogram(runs_df, x="metrics.test_rmse", title="Test RMSE Distribution", nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "metrics.test_r2" in runs_df.columns:
                    fig = px.histogram(runs_df, x="metrics.test_r2", title="Test R² Distribution", nbins=10)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Recent Runs")
            display_cols = [c for c in ["run_id", "metrics.test_rmse", "metrics.test_r2"] if c in runs_df.columns]
            st.dataframe(runs_df[display_cols].head(10), use_container_width=True)

            best_run = mlflow_mgr.get_best_run(metric="test_rmse", ascending=True)
            if best_run:
                st.markdown("### Best Model")
                st.json({
                    "run_id": best_run.get("run_id"),
                    "test_rmse": best_run.get("metrics.test_rmse"),
                    "test_r2": best_run.get("metrics.test_r2")
                })
        else:
            st.info("No MLflow runs found. Train a model first using:")
            st.code("python main.py train")

    with tab2:
        st.markdown("### ONNX Model Management")

        if st.session_state.onnx_predictor:
            model_name = os.path.basename(st.session_state.onnx_predictor.onnx_path)
            st.success(f"✅ ONNX Model Loaded: {model_name}")
        else:
            st.warning("⚠️ No ONNX model loaded")

        onnx_dir = "onnx_models"
        if os.path.exists(onnx_dir):
            onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith('.onnx')]

            if onnx_files:
                st.markdown("### Available Models")
                model_data = []
                for f in sorted(onnx_files):
                    path = os.path.join(onnx_dir, f)
                    model_data.append({"Filename": f, "Size (KB)": f"{os.path.getsize(path)/1024:.1f}"})
                st.dataframe(pd.DataFrame(model_data), use_container_width=True)

                st.markdown("---")
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_onnx = st.selectbox("Select Model", onnx_files, key="onnx_select")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("📦 Load Model", use_container_width=True, type="primary"):
                        onnx_path = os.path.join(onnx_dir, selected_onnx)
                        with st.spinner(f"Loading {selected_onnx}..."):
                            predictor = load_onnx_model(onnx_path)
                            if predictor:
                                st.session_state.onnx_predictor = predictor
                                st.success(f"✅ Model loaded: {selected_onnx}")
                                st.rerun()

                if st.session_state.onnx_predictor:
                    st.markdown("---")
                    if st.button("🏃 Run Benchmark (1000 iterations)"):
                        with st.spinner("Benchmarking..."):
                            bench = st.session_state.onnx_predictor.benchmark(1000)
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Inferences/sec", f"{bench['inferences_per_second']:.0f}")
                            c2.metric("Avg Latency", f"{bench['avg_time_ms']:.3f} ms")
                            c3.metric("Total Time", f"{bench['total_time_seconds']:.3f}s")
                            c4.metric("Iterations", bench['n_iterations'])

                    if st.button("🗑️ Unload Model", type="secondary"):
                        st.session_state.onnx_predictor = None
                        st.session_state.onnx_loaded = False
                        st.success("Model unloaded")
                        st.rerun()
            else:
                st.warning("No ONNX models found.")
                if st.button("📤 Create Model Now"):
                    try:
                        from onnx_export import ONNXExporter
                        exporter = ONNXExporter()
                        onnx_path, _ = exporter.export_best_model()
                        st.success(f"Model created: {onnx_path}")
                        st.session_state.onnx_predictor = ONNXPredictor(onnx_path)
                        st.session_state.onnx_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.warning("'onnx_models/' directory not found.")
            if st.button("Create Directory & Model"):
                os.makedirs("onnx_models", exist_ok=True)
                st.rerun()

    with tab3:
        st.markdown("### Feature Importance Analysis")
        feature_descriptions = {
            "jaccard_similarity": "Medical condition match score",
            "risk_score": "Patient clinical risk level",
            "urgency_score": "Treatment urgency level",
            "expertise_score": "Provider expertise rating",
            "success_rate": "Provider treatment success rate",
            "availability_score": "Provider current availability",
            "inverse_load": "Inverse of current patient load"
        }
        features_df = pd.DataFrame([{"Feature": k, "Description": v} for k, v in feature_descriptions.items()])
        st.dataframe(features_df, use_container_width=True)

        st.markdown("### Scoring Weights")
        weights_df = pd.DataFrame([{"Factor": k.capitalize(), "Weight": v} for k, v in config.SCORE_WEIGHTS.items()])
        fig = px.bar(weights_df, x="Factor", y="Weight", title="Recommendation Score Weights",
                     color="Weight", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Page: Explainability
# ==========================================
elif page == "🔬 Explainability":
    st.markdown('<h1 class="main-header">🔬 Explainability Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Understand **why** the system recommends specific providers using SHAP.")

    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.selectbox("Patient ID", [f"Patient_{i:03d}" for i in range(1, 101)])
    with col2:
        provider_id = st.selectbox("Provider ID", [f"Doctor_{i:03d}" for i in range(1, 31)])

    if st.button("🔬 Generate Explanation", type="primary"):
        with st.spinner("Generating SHAP explanation..."):
            conn = get_connection()
            patient = fetch_patient(conn, patient_id)
            provider_dict = fetch_all_providers(conn)
            provider = next((p for p in provider_dict if p['provider_id'] == provider_id), None)
            conn.close()

            if patient and provider:
                explainer = RecommendationExplainer()
                explanation_df = explainer.explain(patient, provider)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Input Features")
                    features = build_features(patient, provider)
                    features_display = pd.DataFrame({
                        "Feature": config.FEATURE_NAMES,
                        "Value": [f"{v:.4f}" for v in features]
                    })
                    st.dataframe(features_display, use_container_width=True)

                with col2:
                    st.markdown("### SHAP Values")
                    shap_display = explanation_df[["feature", "shap_value", "impact"]]
                    shap_display["shap_value"] = shap_display["shap_value"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(shap_display, use_container_width=True)

                st.markdown("### Feature Impact Visualization")
                shap_records = explanation_df[["feature", "shap_value"]].to_dict("records")
                fig = plot_shap_waterfall(shap_records)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                score = fallback_score(features)
                st.markdown(f"### Final Score: **{score:.4f}**")

                with st.expander("📐 Score Formula Breakdown"):
                    w = config.SCORE_WEIGHTS
                    for label, weight, feat_idx in [
                        ("Jaccard", w['jaccard'], 0), ("Risk", w['risk'], 1),
                        ("Urgency", w['urgency'], 2), ("Expertise", w['expertise'], 3),
                        ("Success", w['success'], 4), ("Availability", w['availability'], 5)
                    ]:
                        st.markdown(f"**{label}**: {weight} × {features[feat_idx]:.4f} = {weight * features[feat_idx]:.4f}")
                    weighted_sum = sum(w[k] * features[i] for i, k in enumerate(['jaccard','risk','urgency','expertise','success','availability']))
                    st.markdown(f"**Weighted Sum**: {weighted_sum:.4f}")
                    st.markdown(f"**Inverse Load**: {features[6]:.4f}")
                    st.markdown(f"**Final Score**: {weighted_sum:.4f} × {features[6]:.4f} = **{score:.4f}**")
            else:
                st.error("Patient or provider not found.")


# ==========================================
# Page: MLflow Dashboard
# ==========================================
# ==========================================
# Page: MLflow Dashboard
# ==========================================
elif page == "📈 MLflow Dashboard":
    st.markdown('<h1 class="main-header">📈 MLflow Dashboard</h1>', unsafe_allow_html=True)
    
    # Get MLflow manager from session state
    mlflow_mgr = st.session_state.mlflow_mgr

    st.markdown("""
    ### MLflow Tracking Server
    Access the full MLflow UI at: [http://localhost:5000](http://localhost:5000)
    ```bash
    python main.py mlflow-ui
    ```
    """)

    st.markdown("---")
    st.markdown("### Experiments")
    experiments = mlflow_mgr.list_experiments()

    if experiments:
        for exp in experiments:
            with st.expander(f"📁 {exp['name']} (ID: {exp['experiment_id']})"):
                st.markdown(f"Location: {exp['artifact_location']}")
                st.markdown(f"Stage: {exp['lifecycle_stage']}")

                runs_df = mlflow_mgr.search_runs(max_results=10)

                if not runs_df.empty:
                    st.markdown(f"Total Runs: {len(runs_df)}")
    else:
        st.info("No experiments found. Train a model first using:")
        st.code("python main.py train")

    st.markdown("---")
    st.markdown("### Registered Models")

    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()

        if registered_models:
            for model in registered_models:
                with st.expander(f"🤖 {model.name}"):
                    versions = client.search_model_versions(f"name='{model.name}'")

                    if versions:
                        version_data = []

                        for v in versions:
                            version_data.append({
                                "Version": v.version,
                                "Stage": v.current_stage,
                                "Run ID": v.run_id[:8] + "...",
                                "Status": v.status
                            })

                        st.dataframe(pd.DataFrame(version_data), use_container_width=True)
        else:
            st.info("No registered models found.")

    except Exception as e:
        st.warning(f"Could not fetch registered models: {e}")

    st.markdown("---")
    st.markdown("### Recent Activity")

    recent_runs = mlflow_mgr.search_runs(max_results=5)

    if not recent_runs.empty:
        for _, run in recent_runs.iterrows():
            run_id_short = run["run_id"][:8] if "run_id" in run else "N/A"
            rmse = run.get("metrics.test_rmse", "N/A")
            r2 = run.get("metrics.test_r2", "N/A")

            # Format numbers safely
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)

            st.markdown(f"""
            **Run:** {run_id_short}...  
            📊 RMSE: {rmse_str} | R²: {r2_str}
            """)
    else:
        st.info("No recent runs found.")
        
# ==========================================
# Footer (runs on every page)
# ==========================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🏥 Healthcare Recommendation System v1.0.0</p>
    <p>Powered by MLflow, ONNX, and Streamlit</p>
</div>
""", unsafe_allow_html=True)