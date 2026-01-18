import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    auc
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="HVAC Failure Detection Dashboard",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UTIL: CREATE 3-LABEL DATASET WITH EARLY WARNING WINDOW
# ============================================================
def create_3label_dataset(df: pd.DataFrame, early_warning_days: int = 15) -> pd.DataFrame:
    """
    Convert binary labels (0, 1) to 3-label (0, 2, 1) with early warning window.
    
    Logic:
    - Label 0: Normal operation
    - Label 1: Failure occurred
    - Label 2: Pre-failure (within N days before a failure)
    
    Args:
        df: DataFrame with columns [timestamp, core_id, label, ...]
        early_warning_days: Days before failure to mark as pre-failure (label=2)
    
    Returns:
        DataFrame with updated 'label' column containing 0, 2, or 1
    """
    df = df.copy()
    df = ensure_timestamp(df)
    
    # Initialize new label column
    df['label_3class'] = df['label'].copy()
    
    # Process each core separately
    for core_id in df['core_id'].unique():
        core_mask = df['core_id'] == core_id
        core_df = df[core_mask].copy()
        
        # Find all failure points (label=1)
        failure_indices = core_df[core_df['label'] == 1].index
        
        if len(failure_indices) == 0:
            continue  # No failures in this core
        
        # For each failure, mark the preceding window as pre-failure (label=2)
        for fail_idx in failure_indices:
            fail_time = core_df.loc[fail_idx, 'timestamp']
            early_warning_start = fail_time - pd.Timedelta(days=early_warning_days)
            
            # Find indices in the early warning window
            warning_mask = (
                (core_df['timestamp'] >= early_warning_start) &
                (core_df['timestamp'] < fail_time) &
                (core_df['label'] == 0)  # Only convert normal points, not existing failures
            )
            
            # Update to label=2 (pre-failure)
            df.loc[core_df[warning_mask].index, 'label_3class'] = 2
    
    # Replace original label with 3-class label
    df['label_original'] = df['label']  # Keep backup
    df['label'] = df['label_3class']
    df = df.drop(columns=['label_3class'])
    
    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_data(parquet_path: str | None):
    """Load data and create both binary and 3-label versions."""
    if parquet_path and os.path.exists(parquet_path):
        df_binary = pd.read_parquet(parquet_path)
        df_binary = ensure_timestamp(df_binary)
        
        # Create 3-label version (15-day pre-failure window)
        df_3label = create_3label_dataset(df_binary.copy(), early_warning_days=15)
        
        return df_binary, df_3label
    return None, None


# ============================================================
# UTIL: PREPROCESS + NORMALIZATION (must match training)
# ============================================================
def preprocess_hvac_mode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode hvacMode column if present."""
    df = df.copy()
    if "hvacMode" in df.columns:
        df["hvacMode"] = df["hvacMode"].astype(str)
        dummies = pd.get_dummies(df["hvacMode"], prefix="hvacMode")
        df = pd.concat([df.drop(columns=["hvacMode"]), dummies], axis=1)
    return df

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column exists and is properly formatted."""
    df = df.copy()

    # If timestamp exists as BOTH index level and column, drop the column and rebuild cleanly
    if "timestamp" in df.columns and (
        (df.index.name == "timestamp") or
        ("timestamp" in getattr(df.index, "names", []))
    ):
        df = df.drop(columns=["timestamp"])

    # If timestamp is in the index (DatetimeIndex or named index), move it to a column
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "timestamp":
        df = df.reset_index()
        if "timestamp" not in df.columns:
            if "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})

    # If timestamp column exists now, parse + sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    else:
        # absolute fallback
        df["timestamp"] = pd.RangeIndex(len(df))

    return df


def apply_per_core_scaling(
    sub_df: pd.DataFrame,
    feature_cols: list,
    core_id: str,
    core_scalers: dict | None
) -> pd.DataFrame:
    """
    Apply per-core scaling ONLY. No global scaler fallback.
    Returns dataframe with scaled features + label/core_id/timestamp kept.
    """
    sub = sub_df.copy()
    keep_cols = [c for c in ["label", "core_id", "timestamp"] if c in sub.columns]

    X = sub[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Apply per-core scaler if available
    if core_scalers is not None and core_id in core_scalers:
        cols, scaler = core_scalers[core_id]
        # Use intersection to avoid column mismatch crashes
        cols_use = [c for c in cols if c in X.columns]
        X_scaled = X.copy()
        if cols_use:  # Only scale if there are matching columns
            X_scaled[cols_use] = scaler.transform(X_scaled[cols_use])
        out = pd.concat([sub[keep_cols], X_scaled], axis=1)
        return out

    # If no scaler available for this core, return unscaled (with warning logged)
    out = pd.concat([sub[keep_cols], X], axis=1)
    return out


# ============================================================
# UTIL: IF + LSTM AE SCORE (for dashboard inference)
# ============================================================
def add_iforest_score(df_scaled: pd.DataFrame, base_features: list, iforest_models: dict) -> pd.DataFrame:
    """Add isolation forest anomaly score to dataframe."""
    df = df_scaled.copy()
    
    if iforest_models is None or len(df) == 0:
        df["iforest_score"] = 0.0
        return df

    cid = str(df["core_id"].iloc[0]) if "core_id" in df.columns else None
    if cid not in iforest_models:
        df["iforest_score"] = 0.0
        return df

    m = iforest_models[cid]

    # Use the exact features the model was trained on
    if hasattr(m, "feature_names_in_"):
        cols = list(m.feature_names_in_)
    else:
        # fallback: assume it was trained on base_features
        cols = list(base_features)

    # Build X with exact columns & order; fill missing with 0; drop extras automatically
    X = df.reindex(columns=cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = -m.decision_function(X)  # higher => more anomalous
    score = np.asarray(score, dtype=float)

    # Normalize within this core slice
    if score.std() > 1e-6:
        score = (score - score.mean()) / score.std()
    
    df["iforest_score"] = score
    return df


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sequences for LSTM autoencoder."""
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=float)
    return np.array([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)], dtype=float)


def add_lstm_ae_score(
    df_scaled: pd.DataFrame,
    base_features: list,
    ae_models: dict,
    seq_len: int
) -> pd.DataFrame:
    """Add LSTM autoencoder reconstruction error score to dataframe."""
    df = df_scaled.copy()
    cid = str(df["core_id"].iloc[0]) if "core_id" in df.columns and len(df) else None

    if ae_models is None or cid not in ae_models:
        df["lstm_ae_score"] = 0.0
        return df

    model = ae_models[cid]

    X = df[base_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    X_seq = make_sequences(X, seq_len)
    if len(X_seq) == 0:
        df["lstm_ae_score"] = 0.0
        return df

    recon = model.predict(X_seq, verbose=0)
    mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

    # Pad to match original length
    padded = np.concatenate([np.repeat(mse[0], seq_len - 1), mse])
    
    # Normalize
    if padded.std() > 1e-6:
        padded = (padded - padded.mean()) / padded.std()

    df["lstm_ae_score"] = padded
    df["lstm_ae_score"] = df["lstm_ae_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ============================================================
# UTIL: FEATURE ALIGNMENT
# ============================================================
def align_to_model_features(model, df: pd.DataFrame, fallback_cols=None) -> pd.DataFrame:
    """
    Align dataframe columns to match model's expected features.
    Fills missing columns with 0, drops extra columns.
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    elif fallback_cols is not None:
        cols = list(fallback_cols)
    else:
        raise ValueError("No feature list available to align input.")
    
    X = df.reindex(columns=cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# ============================================================
# UTIL: PREDICTION HELPERS
# ============================================================
def rf_binary_predict_proba(rf, X: pd.DataFrame) -> np.ndarray:
    """Get probability of positive class for binary RF."""
    proba = rf.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    # Fallback if model was trained differently
    return np.max(proba, axis=1)


def compute_unit_metrics(df: pd.DataFrame, y_pred: np.ndarray, positive_labels=(1,)) -> dict:
    """Compute unit-level (per-core) metrics."""
    tmp = df.copy()
    tmp["pred"] = y_pred

    unit_true = tmp.groupby("core_id")["label"].apply(lambda x: int(any(v in positive_labels for v in x)))
    unit_pred = tmp.groupby("core_id")["pred"].apply(lambda x: int(any(v in positive_labels for v in x)))

    tp = int(((unit_true == 1) & (unit_pred == 1)).sum())
    fp = int(((unit_true == 0) & (unit_pred == 1)).sum())
    fn = int(((unit_true == 1) & (unit_pred == 0)).sum())
    tn = int(((unit_true == 0) & (unit_pred == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1
    }


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_timeline(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None, title: str, is_3label: bool = False):
    """Plot timeline of true vs predicted labels with probability overlay."""
    t = df["timestamp"]
    fig = go.Figure()

    if is_3label:
        # Color mapping for 3-label: 0=green, 2=orange (pre-failure), 1=red (failure)
        color_map = {0: 'rgb(34, 139, 34)', 2: 'rgb(255, 165, 0)', 1: 'rgb(220, 20, 60)'}
        color_map_light = {0: 'rgba(34, 139, 34, 0.2)', 2: 'rgba(255, 165, 0, 0.3)', 1: 'rgba(220, 20, 60, 0.3)'}
        label_names = {0: 'Normal', 2: 'Pre-Failure', 1: 'Failure'}
        
        # Add shaded background regions for true labels
        current_label = y_true[0]
        start_idx = 0
        
        for i in range(1, len(y_true)):
            if y_true[i] != current_label or i == len(y_true) - 1:
                end_idx = i if y_true[i] != current_label else i + 1
                
                # Add shaded region
                fig.add_vrect(
                    x0=t.iloc[start_idx],
                    x1=t.iloc[end_idx - 1],
                    fillcolor=color_map_light[current_label],
                    layer="below",
                    line_width=0,
                )
                
                start_idx = i
                current_label = y_true[i]
        
        # Plot true labels as horizontal lines at their label value
        for label_val in [0, 2, 1]:
            mask = y_true == label_val
            if mask.any():
                # Create segments for continuous lines
                y_line = y_true.copy().astype(float)
                y_line[~mask] = None  # Break line where label changes
                
                fig.add_trace(go.Scatter(
                    x=t,
                    y=y_line,
                    mode="lines",
                    name=f"True {label_names[label_val]}",
                    line=dict(color=color_map[label_val], width=3),
                    connectgaps=False,
                    hovertemplate=f"t=%{{x}}<br>true={label_names[label_val]}<extra></extra>"
                ))
        
        # Plot predicted labels with markers
        for label_val in [0, 2, 1]:
            mask = y_pred == label_val
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=t[mask], 
                    y=y_pred[mask], 
                    mode="markers",
                    name=f"Pred {label_names[label_val]}",
                    marker=dict(
                        size=8, 
                        color=color_map[label_val], 
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f"t=%{{x}}<br>pred={label_names[label_val]}<extra></extra>"
                ))
    else:
        # Binary classification (original code)
        fig.add_trace(go.Scatter(
            x=t, y=y_true, mode="lines+markers", name="True label",
            line=dict(width=2),
            hovertemplate="t=%{x}<br>true=%{y}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=t, y=y_pred, mode="lines+markers", name="Pred label",
            line=dict(width=2),
            hovertemplate="t=%{x}<br>pred=%{y}<extra></extra>"
        ))

    if prob is not None:
        fig.add_trace(go.Scatter(
            x=t, y=prob, mode="lines", name="Prob/Score",
            yaxis="y2",
            line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
            hovertemplate="t=%{x}<br>prob/score=%{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(title="Label / Class", rangemode="tozero"),
        yaxis2=dict(title="Prob/Score", overlaying="y", side="right", rangemode="tozero"),
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_scores(df: pd.DataFrame, score_col: str, title: str):
    """Plot anomaly scores over time."""
    if score_col not in df.columns:
        st.info(f"No column `{score_col}` found for this model/core.")
        return
    fig = px.line(df, x="timestamp", y=score_col, title=title)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, aspect="auto", title=title)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# LOADING (artifacts + data)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data_legacy(parquet_path: str | None):
    """Legacy loader - kept for backward compatibility."""
    if parquet_path and os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        return df
    return None


@st.cache_resource(show_spinner=False)
def load_joblib(path: str):
    """Load joblib artifact."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def safe_listdir(folder: str):
    """Safely list directory contents."""
    try:
        return sorted(os.listdir(folder))
    except Exception:
        return []


# ============================================================
# SIDEBAR: PATHS + MODEL SELECTION
# ============================================================
st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("### Data source")
default_path = "/Users/chiragseth/Desktop/Alert Labs Project/My_Files/Power-Factor-approach/final_df_2.parquet"
parquet_path = st.sidebar.text_input("Parquet path", value=default_path)

df_raw = None
df_raw_3label = None

if os.path.exists(parquet_path):
    with st.spinner("Loading data and creating 3-label version..."):
        df_raw, df_raw_3label = load_and_prepare_data(parquet_path)
    
    if df_raw is not None and df_raw_3label is not None:
        # Show statistics for 3-label dataset
        total_0 = (df_raw_3label['label'] == 0).sum()
        total_2 = (df_raw_3label['label'] == 2).sum()
        total_1 = (df_raw_3label['label'] == 1).sum()
        st.sidebar.success(f"✅ Data loaded! 3-label: 0={total_0:,} | 2={total_2:,} | 1={total_1:,}")
else:
    st.sidebar.warning("Parquet file not found. Please provide a valid path.")

st.sidebar.markdown("### Artifacts folder")
art_dir = st.sidebar.text_input("Artifacts folder", value=".")
files = safe_listdir(art_dir)

def path_in_dir(fname: str) -> str:
    return os.path.join(art_dir, fname)

st.sidebar.markdown("### Model artifacts")
rf_baseline_path = st.sidebar.text_input("Baseline RF (binary)", value="rf_best_recall.joblib")
rf_baseline_feats_path = st.sidebar.text_input("Baseline features", value="rf_feature_columns.joblib")
rf_baseline_scalers_path = st.sidebar.text_input("Baseline core scalers", value="rf_core_scalers.joblib")

rf_if_path = st.sidebar.text_input("IF+RF hybrid (binary)", value="rf_hybrid_best_recall.joblib")
iforest_models_path = st.sidebar.text_input("IF models (binary)", value="iforest_models.joblib")
rf_if_feats_path = st.sidebar.text_input("IF hybrid features", value="rf_feature_columns.joblib")
rf_if_scalers_path = st.sidebar.text_input("IF hybrid core scalers", value="rf_core_scalers.joblib")

rf_lstm_path = st.sidebar.text_input("LSTM+RF hybrid (binary)", value="rf_hybrid_best_recall_1.joblib")
lstm_models_path = st.sidebar.text_input("LSTM AE models", value="lstm_ae_models.joblib")
rf_lstm_feats_path = st.sidebar.text_input("LSTM hybrid features", value="rf_feature_columns_1.joblib")
rf_lstm_scalers_path = st.sidebar.text_input("LSTM hybrid core scalers", value="rf_core_scalers_1.joblib")
seq_len = st.sidebar.number_input("LSTM seq_len", min_value=5, max_value=200, value=30, step=1)

rf_3_path = st.sidebar.text_input("3-label IF+RF", value="rf_hybrid_3label_best_recall.joblib")
iforest_3_path = st.sidebar.text_input("IF models (3-label)", value="iforest_models_3label.joblib")
feats_3_path = st.sidebar.text_input("3-label hybrid features", value="rf_feature_columns_3label.joblib")
scalers_3_path = st.sidebar.text_input("3-label core scalers", value="rf_core_scalers_3label.joblib")

# ⚠️ REMOVED: Global scaler - not needed for per-core normalization

st.sidebar.markdown("### Prediction settings")

# Model-specific thresholds stored in session state
if "threshold_baseline" not in st.session_state:
    st.session_state.threshold_baseline = 0.80
if "threshold_if" not in st.session_state:
    st.session_state.threshold_if = 0.80
if "threshold_lstm" not in st.session_state:
    st.session_state.threshold_lstm = 0.22
if "threshold_3label" not in st.session_state:
    st.session_state.threshold_3label = 0.80

# Threshold selection based on active model
threshold_option = st.sidebar.radio(
    "Threshold mode:",
    ["Per-model defaults", "Custom threshold"],
    index=0
)

if threshold_option == "Custom threshold":
    threshold_custom = st.sidebar.slider("Custom threshold (all models)", 0.05, 0.99, 0.50, 0.01)
    threshold_baseline = threshold_custom
    threshold_if = threshold_custom
    threshold_lstm = threshold_custom
    threshold_3label = threshold_custom
else:
    # Show per-model thresholds
    st.sidebar.markdown("**Per-model thresholds:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        threshold_baseline = st.sidebar.slider("Baseline RF", 0.05, 0.99, st.session_state.threshold_baseline, 0.01, key="slider_baseline")
        threshold_lstm = st.sidebar.slider("LSTM-AE+RF", 0.05, 0.99, st.session_state.threshold_lstm, 0.01, key="slider_lstm")
    with col2:
        threshold_if = st.sidebar.slider("IF+RF", 0.05, 0.99, st.session_state.threshold_if, 0.01, key="slider_if")
        threshold_3label = st.sidebar.slider("3-label IF+RF", 0.05, 0.99, st.session_state.threshold_3label, 0.01, key="slider_3label")
    
    # Update session state
    st.session_state.threshold_baseline = threshold_baseline
    st.session_state.threshold_if = threshold_if
    st.session_state.threshold_lstm = threshold_lstm
    st.session_state.threshold_3label = threshold_3label

rolling = st.sidebar.slider("Rolling smoothing (points)", 1, 200, 1, 1)

# ============================================================
# MAIN UI
# ============================================================
st.title("HVAC Failure Detection — Interactive Model Dashboard")
st.caption("Baseline RF • IF+RF Hybrid • LSTM-AE+RF Hybrid • 3-label IF+RF Hybrid")

if df_raw is None:
    st.warning("No data loaded yet. Use the sidebar to provide a valid parquet path.")
    st.stop()

# Basic checks
req_cols = {"core_id", "label"}
missing = req_cols - set(df_raw.columns)
if missing:
    st.error(f"Your dataframe is missing required columns: {missing}")
    st.stop()

# Choose core - use binary dataset for selection
core_ids = df_raw["core_id"].astype(str).unique().tolist()
core_ids_sorted = sorted(core_ids)

DEFAULT_CORE_ID = "67a0c7ffde52d3c9b2b7923d"

# Find index safely
if DEFAULT_CORE_ID in core_ids_sorted:
    default_index = core_ids_sorted.index(DEFAULT_CORE_ID)
else:
    default_index = 0

selected_core = st.sidebar.selectbox(
    "Select core_id",
    core_ids_sorted,
    index=default_index
)

# Get core data - binary for binary models, 3-label for 3-label model
core_df_raw = df_raw[df_raw["core_id"].astype(str) == str(selected_core)].copy()
core_df_raw = ensure_timestamp(core_df_raw)

core_df_raw_3label = df_raw_3label[df_raw_3label["core_id"].astype(str) == str(selected_core)].copy()
core_df_raw_3label = ensure_timestamp(core_df_raw_3label)

# Preprocessing to match training
df_model = preprocess_hvac_mode(df_raw)
core_df_model = preprocess_hvac_mode(core_df_raw)

df_model_3label = preprocess_hvac_mode(df_raw_3label)
core_df_model_3label = preprocess_hvac_mode(core_df_raw_3label)

# ============================================================
# LOAD MODELS
# ============================================================
rf_baseline = load_joblib(path_in_dir(rf_baseline_path))
baseline_feats = load_joblib(path_in_dir(rf_baseline_feats_path))
baseline_scalers = load_joblib(path_in_dir(rf_baseline_scalers_path))

rf_if = load_joblib(path_in_dir(rf_if_path))
iforest_models = load_joblib(path_in_dir(iforest_models_path))
if_feats = load_joblib(path_in_dir(rf_if_feats_path))
if_scalers = load_joblib(path_in_dir(rf_if_scalers_path))

rf_lstm = load_joblib(path_in_dir(rf_lstm_path))
ae_models = load_joblib(path_in_dir(lstm_models_path))
lstm_feats = load_joblib(path_in_dir(rf_lstm_feats_path))
lstm_scalers = load_joblib(path_in_dir(rf_lstm_scalers_path))

rf_3 = load_joblib(path_in_dir(rf_3_path))
iforest_3 = load_joblib(path_in_dir(iforest_3_path))
feats_3 = load_joblib(path_in_dir(feats_3_path))
scalers_3 = load_joblib(path_in_dir(scalers_3_path))

# ⚠️ REMOVED: No global scaler loading - using per-core only

# ============================================================
# TOP SUMMARY
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (core)", f"{len(core_df_raw):,}")
c2.metric("Binary - Positives", int((core_df_raw["label"] > 0).sum()))
c3.metric("3-Label - Pre-fail (2)", int((core_df_raw_3label["label"] == 2).sum()))
c4.metric("3-Label - Failure (1)", int((core_df_raw_3label["label"] == 1).sum()))

# ============================================================
# MODEL RUNNERS
# ============================================================
def run_baseline_binary():
    """Run baseline RF binary model."""
    if rf_baseline is None or baseline_feats is None:
        return None, None, None, "Baseline artifacts missing."

    feature_cols = [c for c in baseline_feats if c in core_df_model.columns]
    df_scaled = apply_per_core_scaling(
        core_df_model, feature_cols, str(selected_core),
        core_scalers=baseline_scalers
    )
    
    X = align_to_model_features(rf_baseline, df_scaled, fallback_cols=baseline_feats)

    prob = rf_binary_predict_proba(rf_baseline, X)
    if rolling > 1:
        prob = pd.Series(prob).rolling(rolling, min_periods=1).mean().values
    pred = (prob > threshold_baseline).astype(int)

    y_true = core_df_raw["label"].values
    return df_scaled, pred, prob, None


def run_if_hybrid_binary():
    """Run IF + RF hybrid binary model."""
    if rf_if is None or if_feats is None:
        return None, None, None, "IF hybrid artifacts missing."

    # Base features (exclude iforest_score)
    base_features = [c for c in if_feats if c != "iforest_score" and c in core_df_model.columns]

    df_scaled = apply_per_core_scaling(
        core_df_model, base_features, str(selected_core),
        core_scalers=if_scalers
    )

    # Add IF score
    df_h = add_iforest_score(df_scaled, base_features, iforest_models)

    # Align to RF expected features
    X = align_to_model_features(rf_if, df_h, fallback_cols=if_feats)

    prob = rf_binary_predict_proba(rf_if, X)
    if rolling > 1:
        prob = pd.Series(prob).rolling(rolling, min_periods=1).mean().values
    pred = (prob > threshold_if).astype(int)

    return df_h, pred, prob, None


def run_lstm_hybrid_binary():
    """Run LSTM-AE + RF hybrid binary model."""
    if rf_lstm is None or lstm_feats is None:
        return None, None, None, "LSTM hybrid artifacts missing."

    base_features = [c for c in lstm_feats if c != "lstm_ae_score" and c in core_df_model.columns]

    df_scaled = apply_per_core_scaling(
        core_df_model, base_features, str(selected_core),
        core_scalers=lstm_scalers
    )
    
    df_h = add_lstm_ae_score(df_scaled, base_features, ae_models, int(seq_len))
    
    X = align_to_model_features(rf_lstm, df_h, fallback_cols=lstm_feats)

    prob = rf_binary_predict_proba(rf_lstm, X)
    if rolling > 1:
        prob = pd.Series(prob).rolling(rolling, min_periods=1).mean().values
    pred = (prob > threshold_lstm).astype(int)

    return df_h, pred, prob, None


def run_3label_if_hybrid():
    """Run 3-label IF + RF hybrid model - uses 3-label dataset."""
    if rf_3 is None or feats_3 is None:
        return None, None, None, "3-label artifacts missing."

    # Use 3-label dataset instead of binary
    base_features = [c for c in feats_3 if c != "iforest_score" and c in core_df_model_3label.columns]

    df_scaled = apply_per_core_scaling(
        core_df_model_3label, base_features, str(selected_core),
        core_scalers=scalers_3
    )
    
    df_h = add_iforest_score(df_scaled, base_features, iforest_3)
    
    X = align_to_model_features(rf_3, df_h, fallback_cols=feats_3)

    # 3-class: use direct predict
    pred = rf_3.predict(X)
    prob = None
    if hasattr(rf_3, "predict_proba"):
        proba = rf_3.predict_proba(X)
        # Show max probability as a "confidence" line
        prob = np.max(proba, axis=1)
        if rolling > 1:
            prob = pd.Series(prob).rolling(rolling, min_periods=1).mean().values

    return df_h, pred, prob, None


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Core Explorer",
    "Compare Models",
    "Metrics (Core-level)",
    "Data + Export"
])

# ============================================================
# TAB 1: CORE EXPLORER
# ============================================================
with tab1:
    st.subheader(f"Core Explorer — {selected_core}")

    model_choice = st.selectbox(
        "Choose model for this core",
        [
            "Baseline RF (binary)",
            "IF + RF Hybrid (binary)",
            "LSTM-AE + RF Hybrid (binary)",
            "IF + RF Hybrid (3-label: 0/2/1)"
        ],
        index=1
    )

    if model_choice == "Baseline RF (binary)":
        df_used, y_pred, prob, err = run_baseline_binary()
        positive_labels = (1,)
        report_labels = [0, 1]
        report_names = ["Normal (0)", "Failure (1)"]
        title = "Baseline RF — timeline (true vs pred) + probability"
        is_3label = False
    elif model_choice == "IF + RF Hybrid (binary)":
        df_used, y_pred, prob, err = run_if_hybrid_binary()
        positive_labels = (1,)
        report_labels = [0, 1]
        report_names = ["Normal (0)", "Failure (1)"]
        title = "IF + RF Hybrid — timeline (true vs pred) + probability"
        is_3label = False
    elif model_choice == "LSTM-AE + RF Hybrid (binary)":
        df_used, y_pred, prob, err = run_lstm_hybrid_binary()
        positive_labels = (1,)
        report_labels = [0, 1]
        report_names = ["Normal (0)", "Failure (1)"]
        title = "LSTM-AE + RF Hybrid — timeline (true vs pred) + probability"
        is_3label = False
    else:
        df_used, y_pred, prob, err = run_3label_if_hybrid()
        positive_labels = (1, 2)
        report_labels = [0, 2, 1]
        report_names = ["Normal (0)", "Pre-Failure (2)", "Failure (1)"]
        title = "3-label IF + RF Hybrid — timeline (true vs pred) + confidence"
        is_3label = True

    if err:
        st.error(err)
    else:
        # Use appropriate dataset for ground truth
        if is_3label:
            y_true = core_df_raw_3label["label"].values
        else:
            y_true = core_df_raw["label"].values

        plot_timeline(df_used, y_true, y_pred, prob, title=title, is_3label=is_3label)

        cA, cB, cC = st.columns(3)
        with cA:
            if model_choice.startswith("IF + RF"):
                plot_scores(df_used, "iforest_score", "iforest_score over time")
            elif model_choice.startswith("LSTM-AE"):
                plot_scores(df_used, "lstm_ae_score", "lstm_ae_score over time")
            else:
                st.info("No auxiliary anomaly score for baseline RF.")
        with cB:
            plot_confusion(y_true, y_pred, labels=report_labels, title="Confusion Matrix (datapoint-level)")
        with cC:
            # Classification report
            try:
                rep = classification_report(
                    y_true, y_pred,
                    labels=report_labels,
                    target_names=report_names,
                    digits=3,
                    zero_division=0,
                    output_dict=False
                )
                st.code(rep)
            except Exception as e:
                st.warning(f"Could not compute classification report: {e}")

        st.markdown("### Quick KPIs (this core)")
        if model_choice.endswith("(binary)"):
            precision, recall, f1, _ = precision_recall_fscore_support(
                (y_true == 1).astype(int),
                (y_pred == 1).astype(int),
                average="binary",
                zero_division=0
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Precision", f"{precision:.3f}")
            k2.metric("Recall", f"{recall:.3f}")
            k3.metric("F1", f"{f1:.3f}")
            k4.metric("Positive rate (pred)", f"{(y_pred==1).mean():.3f}")
            
            # 🔍 DIAGNOSTIC SECTION
            st.markdown("### 🔍 Diagnostics")
            with st.expander("Click to see diagnostic information", expanded=False):
                diag1, diag2, diag3 = st.columns(3)
                with diag1:
                    st.write("**Probability Stats:**")
                    st.write(f"- Min prob: {prob.min():.4f}")
                    st.write(f"- Max prob: {prob.max():.4f}")
                    st.write(f"- Mean prob: {prob.mean():.4f}")
                    st.write(f"- Median prob: {np.median(prob):.4f}")
                    
                    # Get correct threshold for current model
                    if model_choice == "Baseline RF (binary)":
                        current_threshold = threshold_baseline
                    elif model_choice == "IF + RF Hybrid (binary)":
                        current_threshold = threshold_if
                    elif model_choice == "LSTM-AE + RF Hybrid (binary)":
                        current_threshold = threshold_lstm
                    else:
                        current_threshold = threshold_3label
                    
                    st.write(f"- Current threshold: {current_threshold}")
                    st.write(f"- Prob > threshold: {(prob > current_threshold).sum()}")
                
                with diag2:
                    st.write("**True Labels:**")
                    st.write(f"- Total samples: {len(y_true)}")
                    st.write(f"- Label 0 (normal): {(y_true == 0).sum()}")
                    st.write(f"- Label 1 (failure): {(y_true == 1).sum()}")
                    if (y_true == 1).sum() > 0:
                        st.write(f"- Avg prob when true=1: {prob[y_true == 1].mean():.4f}")
                
                with diag3:
                    st.write("**Scaler Info:**")
                    if model_choice == "Baseline RF (binary)":
                        scaler_dict = baseline_scalers
                    elif model_choice == "IF + RF Hybrid (binary)":
                        scaler_dict = if_scalers
                    else:
                        scaler_dict = lstm_scalers
                    
                    if scaler_dict and str(selected_core) in scaler_dict:
                        st.write(f"✅ Per-core scaler found")
                        cols, scaler = scaler_dict[str(selected_core)]
                        st.write(f"- Scaler columns: {len(cols)}")
                    else:
                        st.write(f"⚠️ No scaler for this core")
                
                # Probability distribution
                st.write("**Probability Distribution:**")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=prob, nbinsx=50, name="Probability"))
                fig_hist.add_vline(x=current_threshold, line_dash="dash", line_color="red", 
                                  annotation_text=f"Threshold={current_threshold}")
                fig_hist.update_layout(
                    title="Distribution of Predicted Probabilities",
                    xaxis_title="Probability",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            if prob is not None:
                st.markdown("### Curves (binary)")
                colL, colR = st.columns(2)

                with colL:
                    pr_p, pr_r, _ = precision_recall_curve((y_true == 1).astype(int), prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pr_r, y=pr_p, mode="lines", name="PR"))
                    fig.update_layout(title="Precision-Recall Curve", height=360, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                with colR:
                    fpr, tpr, _ = roc_curve((y_true == 1).astype(int), prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                    fig.update_layout(title="ROC Curve", height=360, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        else:
            # 3-label
            binary_true = (y_true > 0).astype(int)
            binary_pred = (np.asarray(y_pred) > 0).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                binary_true, binary_pred,
                average="binary",
                zero_division=0
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Early Precision (label>0)", f"{precision:.3f}")
            k2.metric("Early Recall (label>0)", f"{recall:.3f}")
            k3.metric("Early F1 (label>0)", f"{f1:.3f}")
            k4.metric("Positive rate (pred>0)", f"{(binary_pred==1).mean():.3f}")
            
            # 3-label specific insights
            st.markdown("### 3-Label Performance Breakdown")
            
            # First show the critical issue
            if (y_pred == 2).sum() == 0:
                st.error("⚠️ **CRITICAL ISSUE**: Model is NOT predicting any Pre-Failure labels (2)!")
                st.warning("""
                **Why this happens:**
                - Severe class imbalance (label=2 is very rare in training data)
                - Model learns to ignore rare class to maximize overall accuracy
                - Default RF doesn't penalize missing rare classes enough
                
                **You MUST retrain with:**
                1. Class weights: `class_weight='balanced'` in RandomForestClassifier
                2. Or SMOTE/oversampling for label=2
                3. Or focal loss / cost-sensitive learning
                
                **Current behavior:** Model is effectively binary (0 vs 1), defeating the purpose of early warning!
                """)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.write("**Pre-Failure (2) Detection:**")
                pre_fail_true = (y_true == 2).sum()
                pre_fail_pred = (y_pred == 2).sum()
                pre_fail_correct = ((y_true == 2) & (y_pred == 2)).sum()
                st.write(f"- True pre-failures: {pre_fail_true}")
                st.write(f"- Predicted pre-failures: {pre_fail_pred} {'❌' if pre_fail_pred == 0 else '✓'}")
                st.write(f"- Correctly detected: {pre_fail_correct}")
                if pre_fail_true > 0:
                    recall_2 = pre_fail_correct/pre_fail_true
                    st.write(f"- Recall: {recall_2:.3f} {'❌' if recall_2 == 0 else '✓'}")
                else:
                    st.write("- No true pre-failures in this core")
            
            with col_b:
                st.write("**Failure (1) Detection:**")
                fail_true = (y_true == 1).sum()
                fail_pred = (y_pred == 1).sum()
                fail_correct = ((y_true == 1) & (y_pred == 1)).sum()
                st.write(f"- True failures: {fail_true}")
                st.write(f"- Predicted failures: {fail_pred}")
                st.write(f"- Correctly detected: {fail_correct}")
                if fail_true > 0:
                    recall_1 = fail_correct/fail_true
                    st.write(f"- Recall: {recall_1:.3f} {'✓' if recall_1 > 0.5 else '⚠️'}")
            
            with col_c:
                st.write("**Class Distribution:**")
                st.write(f"- Normal (0): {(y_true == 0).sum()} true, {(y_pred == 0).sum()} pred")
                st.write(f"- Pre-fail (2): {(y_true == 2).sum()} true, {(y_pred == 2).sum()} pred")
                st.write(f"- Failure (1): {(y_true == 1).sum()} true, {(y_pred == 1).sum()} pred")
                
                # Show probability distribution for each class
                if prob is not None and hasattr(rf_3, "predict_proba"):
                    st.write("**Model Confidence:**")
                    proba_full = rf_3.predict_proba(align_to_model_features(rf_3, df_used, fallback_cols=feats_3))
                    st.write(f"- Avg P(0): {proba_full[:, 0].mean():.3f}")
                    if proba_full.shape[1] > 2:
                        st.write(f"- Avg P(2): {proba_full[:, 1].mean():.3f} {'❌ Too low!' if proba_full[:, 1].mean() < 0.1 else ''}")
                        st.write(f"- Avg P(1): {proba_full[:, 2].mean():.3f}")
            
            # Add actionable visualization
            st.markdown("### 📊 What the Timeline Shows:")
            st.info("""
            **Legend:**
            - 🟢 **Green dots** (●) at y=0: True normal operation
            - 🟠 **Orange dots** (●) at y=2: True pre-failure state (early warning signal!)
            - 🔴 **Red dots** (●) at y=1: True failure occurred
            - ❌ **X marks**: Model predictions (same color scheme)
            
            **What you WANT to see:**
            - Orange X marks appearing BEFORE red dots (early warning working!)
            - Green predictions when actually green (normal operation)
            
            **What you're CURRENTLY seeing:**
            - Almost no orange X marks = Model not giving early warnings
            - Only green and red predictions = Model acting like binary classifier
            """)
            
            # Show probability distribution for debugging
            if hasattr(rf_3, "predict_proba"):
                st.markdown("### 🔍 Probability Distribution Analysis")
                with st.expander("See why model isn't predicting label=2", expanded=False):
                    proba_full = rf_3.predict_proba(align_to_model_features(rf_3, df_used, fallback_cols=feats_3))
                    
                    fig_proba = go.Figure()
                    fig_proba.add_trace(go.Histogram(x=proba_full[:, 0], name="P(Normal=0)", opacity=0.7, marker_color='green'))
                    if proba_full.shape[1] > 2:
                        fig_proba.add_trace(go.Histogram(x=proba_full[:, 1], name="P(Pre-fail=2)", opacity=0.7, marker_color='orange'))
                        fig_proba.add_trace(go.Histogram(x=proba_full[:, 2], name="P(Failure=1)", opacity=0.7, marker_color='red'))
                    
                    fig_proba.update_layout(
                        title="Distribution of Predicted Probabilities for Each Class",
                        xaxis_title="Probability",
                        yaxis_title="Count",
                        barmode='overlay',
                        height=400
                    )
                    st.plotly_chart(fig_proba, use_container_width=True)
                    
                    st.write("**Interpretation:**")
                    st.write("- If orange bars (P(Pre-fail=2)) are all near 0, model never considers pre-failure likely")
                    st.write("- If green and red bars dominate, model is binary (ignoring middle class)")
                    st.write("- **Solution**: Retrain with class_weight='balanced' or custom weights")


# ============================================================
# TAB 2: COMPARE MODELS (SIDE BY SIDE)
# ============================================================
with tab2:
    st.subheader("Compare Models on the Same Core")
    st.caption("Side-by-side timelines so you can visually validate early warnings and false positives.")

    col1, col2 = st.columns(2)

    # Run all 4
    b_df, b_pred, b_prob, b_err = run_baseline_binary()
    i_df, i_pred, i_prob, i_err = run_if_hybrid_binary()
    l_df, l_pred, l_prob, l_err = run_lstm_hybrid_binary()
    t_df, t_pred, t_prob, t_err = run_3label_if_hybrid()

    y_true = core_df_raw["label"].values
    y_true_3label = core_df_raw_3label["label"].values

    with col1:
        if b_err: st.error(b_err)
        else: plot_timeline(b_df, y_true, b_pred, b_prob, "Baseline RF (binary)", is_3label=False)
        if i_err: st.error(i_err)
        else: plot_timeline(i_df, y_true, i_pred, i_prob, "IF + RF Hybrid (binary)", is_3label=False)

    with col2:
        if l_err: st.error(l_err)
        else: plot_timeline(l_df, y_true, l_pred, l_prob, "LSTM-AE + RF Hybrid (binary)", is_3label=False)
        if t_err: st.error(t_err)
        else: plot_timeline(t_df, y_true_3label, t_pred, t_prob, "IF + RF Hybrid (3-label)", is_3label=True)


# ============================================================
# TAB 3: METRICS (DATASET-WIDE)
# ============================================================
with tab3:
    st.subheader("Metrics (dataset-wide, interactive sampling)")
    st.caption("This tab can be heavy on large datasets. You can sample rows or cores.")

    st.markdown("### Choose model to evaluate")
    eval_choice = st.selectbox(
        "Model",
        [
            "Baseline RF (binary)",
            "IF + RF Hybrid (binary)",
            "LSTM-AE + RF Hybrid (binary)",
            "IF + RF Hybrid (3-label)"
        ],
        index=1,
        key="eval_model_choice"  # Add unique key
    )

    sample_mode = st.radio("Evaluate on:", ["This core only", "Sample N cores"], index=0, horizontal=True)
    if sample_mode == "Sample N cores":
        ncores = st.number_input("How many cores to sample?", min_value=2, max_value=min(200, len(core_ids_sorted)), value=min(20, len(core_ids_sorted)), step=1)
        sampled_cores = list(np.random.choice(core_ids_sorted, size=int(ncores), replace=False))
        
        # Use appropriate dataset based on model type
        if eval_choice == "IF + RF Hybrid (3-label)":
            df_eval = df_raw_3label[df_raw_3label["core_id"].astype(str).isin(sampled_cores)].copy()
        else:
            df_eval = df_raw[df_raw["core_id"].astype(str).isin(sampled_cores)].copy()
            
        df_eval = ensure_timestamp(df_eval)
        df_eval_model = preprocess_hvac_mode(df_eval)
    else:
        # Use appropriate dataset for single core
        if eval_choice == "IF + RF Hybrid (3-label)":
            df_eval = core_df_raw_3label.copy()
            df_eval_model = core_df_model_3label.copy()
        else:
            df_eval = core_df_raw.copy()
            df_eval_model = core_df_model.copy()

    st.markdown("### Choose model to evaluate")
    eval_choice = st.selectbox(
        "Model",
        [
            "Baseline RF (binary)",
            "IF + RF Hybrid (binary)",
            "LSTM-AE + RF Hybrid (binary)",
            "IF + RF Hybrid (3-label)"
        ],
        index=1
    )

    # Evaluate per core to respect per-core scalers + per-core anomaly models
    def eval_model_over_df():
        preds_all = []
        probs_all = []
        trues_all = []
        stitched = []

        for cid, sub in df_eval_model.groupby(df_eval["core_id"].astype(str)):
            sub_raw = df_eval[df_eval["core_id"].astype(str) == str(cid)].copy()
            sub = ensure_timestamp(sub.copy())
            sub_raw = ensure_timestamp(sub_raw.copy())

            # Attach core_id
            sub["core_id"] = str(cid)
            sub_raw["core_id"] = str(cid)

            if eval_choice == "Baseline RF (binary)":
                if rf_baseline is None or baseline_feats is None:
                    continue
                feature_cols = [c for c in baseline_feats if c in sub.columns]
                df_scaled = apply_per_core_scaling(sub, feature_cols, str(cid), baseline_scalers)
                X = align_to_model_features(rf_baseline, df_scaled, fallback_cols=baseline_feats)
                prob = rf_binary_predict_proba(rf_baseline, X)
                pred = (prob > threshold_baseline).astype(int)

            elif eval_choice == "IF + RF Hybrid (binary)":
                if rf_if is None or if_feats is None:
                    continue
                base_features = [c for c in if_feats if c != "iforest_score" and c in sub.columns]
                df_scaled = apply_per_core_scaling(sub, base_features, str(cid), if_scalers)
                df_h = add_iforest_score(df_scaled, base_features, iforest_models)
                X = align_to_model_features(rf_if, df_h, fallback_cols=if_feats)
                prob = rf_binary_predict_proba(rf_if, X)
                pred = (prob > threshold_if).astype(int)
                df_scaled = df_h

            elif eval_choice == "LSTM-AE + RF Hybrid (binary)":
                if rf_lstm is None or lstm_feats is None:
                    continue
                base_features = [c for c in lstm_feats if c != "lstm_ae_score" and c in sub.columns]
                df_scaled = apply_per_core_scaling(sub, base_features, str(cid), lstm_scalers)
                df_h = add_lstm_ae_score(df_scaled, base_features, ae_models, int(seq_len))
                X = align_to_model_features(rf_lstm, df_h, fallback_cols=lstm_feats)
                prob = rf_binary_predict_proba(rf_lstm, X)
                pred = (prob > threshold_lstm).astype(int)
                df_scaled = df_h

            else:
                # 3-label
                if rf_3 is None or feats_3 is None:
                    continue
                base_features = [c for c in feats_3 if c != "iforest_score" and c in sub.columns]
                df_scaled = apply_per_core_scaling(sub, base_features, str(cid), scalers_3)
                df_h = add_iforest_score(df_scaled, base_features, iforest_3)
                X = align_to_model_features(rf_3, df_h, fallback_cols=feats_3)
                pred = rf_3.predict(X)
                prob = None
                if hasattr(rf_3, "predict_proba"):
                    proba = rf_3.predict_proba(X)
                    prob = np.max(proba, axis=1)
                df_scaled = df_h

            y_true = sub_raw["label"].values

            preds_all.append(pred)
            trues_all.append(y_true)
            if prob is not None:
                probs_all.append(prob)

            # Stitch exportable frame
            out = pd.DataFrame({
                "timestamp": sub_raw["timestamp"].values,
                "core_id": str(cid),
                "y_true": y_true,
                "y_pred": pred
            })
            if prob is not None:
                out["prob_or_conf"] = prob
            stitched.append(out)

        if not stitched:
            return None, None, None, None

        y_true_all = np.concatenate(trues_all)
        y_pred_all = np.concatenate(preds_all)
        prob_all = np.concatenate(probs_all) if probs_all else None
        export_df = pd.concat(stitched, ignore_index=True)
        return y_true_all, y_pred_all, prob_all, export_df

    with st.spinner("Evaluating..."):
        y_true_all, y_pred_all, prob_all, export_df = eval_model_over_df()

    if export_df is None:
        st.error("Could not evaluate (missing artifacts or empty selection).")
    else:
        if eval_choice.endswith("(3-label)"):
            st.markdown("### Datapoint report (3-label)")
            st.code(classification_report(
                y_true_all, y_pred_all,
                labels=[0, 2, 1],
                target_names=["Normal (0)", "Pre-Failure (2)", "Failure (1)"],
                digits=3,
                zero_division=0
            ))

            st.markdown("### Early-warning binary view (label>0)")
            bt = (y_true_all > 0).astype(int)
            bp = (y_pred_all > 0).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(bt, bp, average="binary", zero_division=0)
            k1, k2, k3 = st.columns(3)
            k1.metric("Precision", f"{precision:.3f}")
            k2.metric("Recall", f"{recall:.3f}")
            k3.metric("F1", f"{f1:.3f}")

        else:
            st.markdown("### Datapoint report (binary)")
            st.code(classification_report(
                y_true_all, y_pred_all,
                labels=[0, 1],
                target_names=["Normal (0)", "Failure (1)"],
                digits=3,
                zero_division=0
            ))

            precision, recall, f1, _ = precision_recall_fscore_support(
                (y_true_all == 1).astype(int),
                (y_pred_all == 1).astype(int),
                average="binary",
                zero_division=0
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Precision", f"{precision:.3f}")
            k2.metric("Recall", f"{recall:.3f}")
            k3.metric("F1", f"{f1:.3f}")

        st.markdown("### Unit metrics (per core)")
        if eval_choice.endswith("(3-label)"):
            unit = compute_unit_metrics(export_df.rename(columns={"y_true": "label", "y_pred": "pred"}), export_df["y_pred"].values, positive_labels=(1, 2))
        else:
            unit = compute_unit_metrics(export_df.rename(columns={"y_true": "label", "y_pred": "pred"}), export_df["y_pred"].values, positive_labels=(1,))

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Unit Precision", f"{unit['precision']:.3f}")
        u2.metric("Unit Recall", f"{unit['recall']:.3f}")
        u3.metric("Unit F1", f"{unit['f1']:.3f}")
        u4.metric("Units flagged (pred)", int((export_df.groupby("core_id")["y_pred"].max() > 0).sum()))

        st.markdown("### Download evaluated predictions")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="dashboard_predictions.csv", mime="text/csv")


# ============================================================
# TAB 4: DATA + EXPORT (CORE)
# ============================================================
with tab4:
    st.subheader("Core data preview + export")

    st.markdown("### Raw core rows")
    st.dataframe(core_df_raw.head(200), use_container_width=True, height=360)

    st.markdown("### Export Options")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**Export current core (raw):**")
        raw_csv = core_df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download core raw CSV", 
            data=raw_csv, 
            file_name=f"{selected_core}_raw.csv", 
            mime="text/csv"
        )
    
    with col_exp2:
        st.markdown("**Export 3-label dataset for training:**")
        st.info("3-label dataset with 15-day pre-failure window is pre-loaded")
        
        # Offer download of pre-generated 3-label dataset
        export_csv = df_raw_3label.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download 3-label dataset (full)",
            data=export_csv,
            file_name="dataset_3label_15day_window.csv",
            mime="text/csv",
            key="export_3label"
        )
        
        st.caption("💡 Use this to retrain your 3-label model")
    
  

    st.markdown("### Notes / Common gotchas")
    st.info(
        "1) ✅ All models now use PER-CORE normalization only (no global scaler fallback).\n"
        "2) Make sure your dashboard uses the same preprocessing as training (hvacMode OHE + per-core scalers).\n"
        "3) LSTM AE models saved via joblib can be fragile across environments; if you hit load errors, save Keras models with model.save().\n"
        "4) Feature alignment ensures models only see columns they were trained on (drops extras, fills missing with 0)."
    )
    

    