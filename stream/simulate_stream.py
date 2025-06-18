import time
import pandas as pd
import numpy as np
from datetime import datetime

def generate_row():
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_util": np.random.normal(50, 10),
        "memory_util": np.random.normal(60, 5),
        "network_io": np.random.normal(100, 20),
        "disk_io": np.random.normal(70, 15)
    }

def simulate_stream(file_path, interval=2):
    print(f"Starting stream to {file_path}...")
    while True:
        row = generate_row()
        df = pd.DataFrame([row])
        df.to_csv(file_path, mode='a', index=False, header=not bool(open(file_path).read()))
        time.sleep(interval)

if __name__ == "__main__":
    simulate_stream("data/stream.csv", interval=2)
