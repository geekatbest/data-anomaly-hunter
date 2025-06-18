from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.05)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        preds = self.model.predict(data)
        return pd.Series(preds).map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly → map to 0/1
