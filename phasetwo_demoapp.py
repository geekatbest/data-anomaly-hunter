import streamlit as st
import pandas as pd
from utils.prophet_detector import ProphetAnomalyDetector
from utils.rule_engine import RuleEngine

st.set_page_config(layout="wide")
st.title("🧪 Phase 2 — Time-Series + Rule Engine Test")

# Load data
df = pd.read_csv("data/stream.csv").dropna()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.tail(200)

# Apply Prophet Anomaly Detection
prophet_detector = ProphetAnomalyDetector(threshold=10)
df_prophet = prophet_detector.fit_predict(df.copy(), column='cpu_util')

# Apply Rule Engine
rule_engine = RuleEngine(rule_file='rules/rules.yaml')
df_rules = rule_engine.apply_rules(df_prophet)

# Display charts
st.subheader("📈 CPU Utilization & Prophet Prediction")
st.line_chart(df_rules.set_index('timestamp')[['cpu_util', 'yhat']])

st.subheader("🧨 Anomalies (Model + Rules)")
st.dataframe(
    df_rules[['timestamp', 'cpu_util', 'yhat', 'anomaly', 'rule_label', 'rule_anomaly']]
    .style.applymap(lambda x: "background-color: red" if x == 1 else "", subset=['anomaly', 'rule_anomaly'])
)
