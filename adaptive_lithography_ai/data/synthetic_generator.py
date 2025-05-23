import numpy as np
import pandas as pd
import os

def generate_synthetic_data(samples=5000, seed=42):
    np.random.seed(seed)
    data = {
        "exposure_dose": np.random.uniform(10, 100, samples),
        "focus_offset": np.random.normal(0, 1, samples),
        "line_edge_roughness": np.random.uniform(0, 5, samples),
        "critical_dimension": np.random.uniform(30, 90, samples),
        "defect_density": np.random.poisson(2, samples),
    }
    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_litho.csv", index=False)
    print(f"✅ Generated {samples} synthetic samples in data/raw/synthetic_litho.csv")

if __name__ == "__main__":
    generate_synthetic_data()
