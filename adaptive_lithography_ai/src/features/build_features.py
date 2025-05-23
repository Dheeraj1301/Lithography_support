import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dose_focus_ratio"] = df["exposure_dose"] / (df["focus_offset"] + 1e-5)
    df["defect_per_dose"] = df["defect_density"] / df["exposure_dose"]
    return df
