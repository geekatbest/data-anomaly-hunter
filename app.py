import streamlit as st
import pandas as pd
import time
from utils.anomaly_detector import AnomalyDetector

st.set_page_config(layout="wide")
st.title("📉 Real-Time Data Anomaly Hunter")

N_ROWS = 100
DATA_FILE = "data/stream.csv"
detector = AnomalyDetector()
model_fitted = False  # Global flag to track training status

# Main loop
placeholder = st.empty()

while True:
    try:
        df = pd.read_csv(DATA_FILE).dropna()

        if len(df) >= 50 and not model_fitted:
            # Fitting on initial batch of 50 rows
            X_init = df[['cpu_util', 'memory_util', 'network_io', 'disk_io']]
            detector.fit(X_init)
            model_fitted = True
            st.success("✅ Model trained on initial 50 rows.")

        if model_fitted:
            df = df.tail(N_ROWS)
            df['anomaly'] = detector.predict(df[['cpu_util', 'memory_util', 'network_io', 'disk_io']])
        else:
            df['anomaly'] = 0  # No prediction until model is trained

        # Charting and Data Display
        with placeholder.container():
            st.line_chart(df.set_index('timestamp')[['cpu_util', 'memory_util']])
            st.dataframe(
                df.tail(N_ROWS).style.applymap(
                    lambda val: "background-color: red" if val == 1 else "",
                    subset=['anomaly']
                )
            )
        time.sleep(3)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
        time.sleep(5)
