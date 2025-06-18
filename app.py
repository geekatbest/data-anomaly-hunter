import streamlit as st
import pandas as pd
import time
from utils.anomaly_detector import AnomalyDetector

st.title("📉 Real-Time Data Anomaly Hunter")

N_ROWS = 100
DATA_FILE = "data/stream.csv"

detector = AnomalyDetector()

@st.cache_resource
def load_and_train():
    # Use initial data to train
    df = pd.read_csv(DATA_FILE)
    X = df[['cpu_util', 'memory_util', 'network_io', 'disk_io']].dropna()
    detector.fit(X)

load_and_train()

placeholder = st.empty()

while True:
    df = pd.read_csv(DATA_FILE).dropna().tail(N_ROWS)
    df['anomaly'] = detector.predict(df[['cpu_util', 'memory_util', 'network_io', 'disk_io']])
    
    with placeholder.container():
        st.line_chart(df.set_index('timestamp')[['cpu_util', 'memory_util']])
        st.dataframe(df.style.applymap(lambda x: "background-color: red" if x == 1 else "", subset=['anomaly']))
    
    time.sleep(3)
