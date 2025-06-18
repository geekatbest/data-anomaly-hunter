from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, df):
        preds = self.model.predict(df[['cpu_util', 'memory_util', 'network_io', 'disk_io']])
        df['anomaly'] = (preds == -1).astype(int)
        return df
