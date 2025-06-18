import streamlit as st
import pandas as pd
import yaml
import json

from utils.anomaly_detector import AnomalyDetector
from utils.prophet_detector import ProphetAnomalyDetector
from utils.rule_engine import RuleEngine

# --- Streamlit Setup ---
st.set_page_config(page_title="🧭 Data Anomaly Hunter", layout="wide")
st.title("🧭 Data Anomaly Hunter Dashboard")

# --- Load Data ---
df = pd.read_csv("data/stream.csv").dropna()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.tail(200)

# --- Sidebar: Rule Engine Editor ---
st.sidebar.header("🧠 Rule Engine Editor")
with open('rules/rules.yaml', 'r') as f:
    rules_data = yaml.safe_load(f)['rules']

editable_rules = []
for rule in rules_data:
    try:
        metric, op, val = rule['if'].split()
        val = float(val)
        label = rule['label']
        new_val = st.sidebar.slider(f"{label} ({rule['if']})", 0.0, 100.0, val, step=1.0)
        editable_rules.append({'if': f"{metric} {op} {new_val}", 'label': label})
    except Exception as e:
        st.sidebar.error(f"Rule parse error: {rule['if']} — {e}")

if st.sidebar.button("💾 Save Rules"):
    with open('rules/rules.yaml', 'w') as f:
        yaml.dump({'rules': editable_rules}, f)
    st.sidebar.success("✅ Rules saved successfully!")

# --- Sidebar: Model Toggle ---
st.sidebar.header("🔀 Choose Detection Method")
model_option = st.sidebar.radio(
    "Select model to run:",
    ("Isolation Forest", "Prophet", "Both")
)

# --- Rule Engine Setup ---
engine = RuleEngine()
engine.set_rules(editable_rules)

# --- Output DF Init ---
df_final = df.copy()

# --- Isolation Forest ---
if model_option in ("Isolation Forest", "Both"):
    st.subheader("🌲 Isolation Forest Results")
    detector = AnomalyDetector()
    detector.fit(df[['cpu_util', 'memory_util', 'network_io', 'disk_io']])
    df_iforest = detector.predict(df.copy())
    df_final['iforest_anomaly'] = df_iforest['anomaly']

# --- Prophet ---
if model_option in ("Prophet", "Both"):
    st.subheader("📈 Prophet Time Series (CPU)")
    threshold = st.slider("Prophet Threshold (abs error)", 1, 50, 10)
    prophet = ProphetAnomalyDetector(threshold=threshold)
    df_prophet = prophet.fit_predict(df.copy(), column='cpu_util')
    df_final['yhat'] = df_prophet['yhat']
    df_final['prophet_anomaly'] = df_prophet['anomaly']

# --- Apply Rules ---
df_final = engine.apply_rules(df_final)

# --- Prophet Forecast Plot ---
if model_option in ("Prophet", "Both"):
    st.markdown("### 📊 CPU Utilization + Prophet Forecast")
    st.line_chart(df_final.set_index('timestamp')[['cpu_util', 'yhat']])

# --- Anomaly Table Display ---
st.markdown("### 🧨 Anomalies Detected")
cols = ['timestamp', 'cpu_util', 'rule_label', 'rule_anomaly']
if 'iforest_anomaly' in df_final.columns:
    cols.append('iforest_anomaly')
if 'prophet_anomaly' in df_final.columns:
    cols.append('prophet_anomaly')

st.dataframe(
    df_final[cols].style.applymap(
        lambda x: "background-color: red" if x == 1 else "",
        subset=['rule_anomaly'] +
        (['iforest_anomaly'] if 'iforest_anomaly' in df_final else []) +
        (['prophet_anomaly'] if 'prophet_anomaly' in df_final else [])
    ),
    use_container_width=True
)

# --- Divider ---
st.markdown("---")

# --- Export Section ---
st.subheader("📤 Export Detected Anomalies")

export_format = st.selectbox("Select export format", ["CSV", "JSON"])
export_only_anomalies = st.checkbox("Export only rows with any anomaly", value=True)

# Filter if required
export_df = df_final.copy()
if export_only_anomalies:
    export_df = export_df[
        (export_df.get('rule_anomaly', 0) == 1) |
        (export_df.get('iforest_anomaly', 0) == 1) |
        (export_df.get('prophet_anomaly', 0) == 1)
    ]

# Download buttons
if export_format == "CSV":
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download as CSV", data=csv, file_name="anomalies.csv", mime="text/csv")
else:
    json_data = export_df.to_json(orient="records")
    st.download_button("⬇️ Download as JSON", data=json_data, file_name="anomalies.json", mime="application/json")
