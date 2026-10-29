from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    APP_DIR / "data/processed/"
    "electricity_forecasting_features.csv"
)

MODEL_PATH = (
    APP_DIR / "recovery/models/"
    "recovery_random_forest.joblib"
)

ANOMALY_PATH = (
    APP_DIR / "recovery/results/anomaly_detection/"
    "recovery_test_anomaly_results.csv"
)

GROUPS_PATH = (
    APP_DIR / "recovery/results/anomaly_detection/"
    "recovery_test_consecutive_anomaly_groups.csv"
)

st.set_page_config(
    page_title="SmartEnergy AI",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ SmartEnergy AI")
st.caption(
    "Hourly electricity forecasting and potential "
    "anomaly exploration"
)

# --------------------------------------------------
# Load existing artifacts
# --------------------------------------------------
@st.cache_data
def load_features(path, modified_time):
    return pd.read_csv(path, parse_dates=["datetime"])

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)

@st.cache_data
def load_anomalies(path, modified_time):
    return pd.read_csv(path, parse_dates=["datetime"])

@st.cache_data
def load_groups(path, modified_time):
    return pd.read_csv(path)

required_paths = [DATA_PATH, MODEL_PATH, ANOMALY_PATH]

missing_paths = [
    str(path) for path in required_paths if not path.is_file()
]

if missing_paths:
    st.error("Required project files are missing:")
    for path in missing_paths:
        st.write(f"- {path}")
    st.stop()

try:
    features = load_features(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
    model = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
    anomalies = load_anomalies(
        str(ANOMALY_PATH), ANOMALY_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load project artifacts: {exc}")
    st.stop()

# --------------------------------------------------
# Validate required columns
# --------------------------------------------------
if "mean_power_kw" not in features.columns:
    st.error("Target column mean_power_kw was not found.")
    st.stop()

if "datetime" not in features.columns:
    st.error("The features dataset has no datetime column.")
    st.stop()

feature_names = list(
    getattr(model, "feature_names_in_", [])
)

if not feature_names:
    st.error(
        "The saved model does not provide feature names. "
        "Inspect its metadata before using this dashboard."
    )
    st.stop()

missing_features = [
    col for col in feature_names if col not in features.columns
]

if missing_features:
    st.error(f"Model input columns missing: {missing_features}")
    st.stop()

features = features.sort_values("datetime").reset_index(drop=True)

# --------------------------------------------------
# Overview
# --------------------------------------------------
st.header("Energy Overview")

latest_time = features["datetime"].max()
average_power = features["mean_power_kw"].mean()
max_power = features["mean_power_kw"].max()

flag_column = "is_potential_anomaly"
flagged_count = None

if flag_column in anomalies.columns:
    flags = (
        anomalies[flag_column]
        .astype(str).str.strip().str.lower()
        .map({"true": True, "false": False})
    )
    flagged_count = int(flags.sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Feature rows", f"{len(features):,}")
col2.metric("Average consumption", f"{average_power:.3f} kW")
col3.metric("Maximum consumption", f"{max_power:.3f} kW")
col4.metric(
    "Potential test anomalies",
    f"{flagged_count:,}" if flagged_count is not None else "N/A"
)

st.caption(
    f"Latest timestamp in the feature dataset: "
    f"{latest_time}"
)

# --------------------------------------------------
# Historical consumption
# --------------------------------------------------
st.header("Historical Electricity Consumption")

days_to_show = st.slider(
    "Hours to display",
    min_value=24,
    max_value=720,
    value=168,
    step=24
)

recent = features.tail(days_to_show)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    recent["datetime"],
    recent["mean_power_kw"],
    linewidth=1
)
ax.set_xlabel("Time")
ax.set_ylabel("Mean active power (kW)")
ax.set_title("Historical hourly electricity consumption")
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Saved model estimate
# --------------------------------------------------
st.header("Saved Model Estimate")

st.write(
    "Select a historical row to calculate the model's estimate "
    "for that row's feature values. This is a historical estimate, "
    "not a future forecast."
)

row_number = st.slider(
    "Historical feature row",
    min_value=0,
    max_value=len(features) - 1,
    value=len(features) - 1
)

selected = features.iloc[[row_number]]
X = selected[feature_names]

try:
    prediction = float(model.predict(X)[0])
    actual = float(selected["mean_power_kw"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Model estimate", f"{prediction:.3f} kW")
    c2.metric("Actual consumption", f"{actual:.3f} kW")
    c3.metric("Absolute error", f"{abs(actual - prediction):.3f} kW")

    st.caption(
        f"Selected timestamp: {selected['datetime'].iloc[0]}"
    )
except Exception as exc:
    st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# Potential anomaly explorer
# --------------------------------------------------
st.header("Potential Anomaly Explorer")

if flag_column not in anomalies.columns:
    st.warning(
        "The anomaly file does not contain the expected flag column."
    )
else:
    anomalies[flag_column] = (
        anomalies[flag_column]
        .astype(str).str.strip().str.lower()
        .map({"true": True, "false": False})
    )

    flagged = anomalies[anomalies[flag_column] == True].copy()

    if "anomaly_direction" in flagged.columns:
        directions = ["All"] + sorted(
            flagged["anomaly_direction"].dropna().unique().tolist()
        )
        selected_direction = st.selectbox(
            "Filter by direction", directions
        )

        if selected_direction != "All":
            flagged = flagged[
                flagged["anomaly_direction"] == selected_direction
            ]

    st.write(f"Matching potential anomalies: {len(flagged):,}")
    st.dataframe(
        flagged.sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered anomalies as CSV",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="smartenergy_filtered_anomalies.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Consecutive alert groups
# --------------------------------------------------
st.header("Consecutive Anomaly Groups")

if GROUPS_PATH.is_file():
    try:
        groups = load_groups(
            str(GROUPS_PATH), GROUPS_PATH.stat().st_mtime
        )
        st.write(f"Saved groups: {len(groups):,}")
        st.dataframe(
            groups,
            use_container_width=True,
            hide_index=True
        )
    except Exception as exc:
        st.warning(f"Could not load anomaly groups: {exc}")
else:
    st.info("The consecutive anomaly groups file is not available.")

st.divider()
st.caption(
    "SmartEnergy AI research prototype. Potential anomalies "
    "indicate unusual prediction errors; they do not confirm "
    "electrical faults."
)
