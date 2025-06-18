import streamlit as st
import pandas as pd
from utils.prophet_detector import ProphetAnomalyDetector
from utils.rule_engine import RuleEngine
import yaml

#Streamlit Setup 
st.set_page_config(page_title="Phase 2 - Anomaly Rules & Time Series", layout="wide")
st.title("📡 Phase 2: Prophet + Editable Rule Engine")

DATA_PATH = "data/stream.csv"
df = pd.read_csv(DATA_PATH).dropna()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.tail(200)  # Limit to recent data for performance

st.subheader("📈 Time-Series Anomaly Detection (Prophet)")
prophet_column = 'cpu_util'
threshold = st.slider("🔧 Anomaly Threshold for Prophet (absolute deviation)", 1, 50, 10)

prophet_detector = ProphetAnomalyDetector(threshold=threshold)
df_prophet = prophet_detector.fit_predict(df.copy(), column=prophet_column)


st.sidebar.header("🧠 Rule Engine Editor")

RULE_PATH = "rules/rules.yaml"
with open(RULE_PATH, 'r') as f:
    rules_data = yaml.safe_load(f)['rules']

editable_rules = []
for rule in rules_data:
    try:
        metric, operator, value = rule['if'].split()
        value = float(value)
        label = rule['label']

        slider_val = st.sidebar.slider(
            f"{label} ({rule['if']})",
            min_value=0.0,
            max_value=100.0,
            value=value,
            step=1.0
        )

        new_rule = {
            'if': f"{metric} {operator} {slider_val}",
            'label': label
        }
        editable_rules.append(new_rule)
    except Exception as e:
        st.sidebar.error(f"Failed to parse rule: {rule['if']} — {e}")


rule_engine = RuleEngine()
rule_engine.set_rules(editable_rules)
df_result = rule_engine.apply_rules(df_prophet)


st.markdown("### 📊 CPU Utilization & Prediction (Prophet)")
st.line_chart(df_result.set_index('timestamp')[['cpu_util', 'yhat']])

st.markdown("### 🧨 Anomalies Detected")
st.dataframe(
    df_result[['timestamp', 'cpu_util', 'yhat', 'anomaly', 'rule_label', 'rule_anomaly']]
    .style.applymap(lambda x: "background-color: red" if x == 1 else "", subset=['anomaly', 'rule_anomaly'])
)


if st.sidebar.button("💾 Save Modified Rules to YAML"):
    with open(RULE_PATH, 'w') as f:
        yaml.dump({'rules': editable_rules}, f)
    st.sidebar.success("Rules updated in rules.yaml")
