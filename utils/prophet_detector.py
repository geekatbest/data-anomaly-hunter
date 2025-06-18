from prophet import Prophet
import pandas as pd

class ProphetAnomalyDetector:
    def __init__(self, threshold=10.0):
        self.threshold = threshold  #error threshold
        self.model = Prophet()

    def fit_predict(self, df, column='cpu_util'):
        df_ts = df[['timestamp', column]].rename(columns={'timestamp': 'ds', column: 'y'}).copy()
        df_ts['ds'] = pd.to_datetime(df_ts['ds'])

        model = Prophet()
        model.fit(df_ts)

        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)

        # Compute anomaly
        df_ts['yhat'] = forecast['yhat']
        df_ts['anomaly'] = (abs(df_ts['y'] - df_ts['yhat']) > self.threshold).astype(int)

        # Merge back with full data
        df_final = df.copy()
        df_final['yhat'] = df_ts['yhat'].values
        df_final['anomaly'] = df_ts['anomaly'].values

        return df_final