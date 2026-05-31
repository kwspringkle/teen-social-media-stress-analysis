from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "raw"


def load_dataset_1() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "Teen_Mental_Health_Dataset.csv")
    df = df.copy()
    df["depression_label"] = df["depression_label"].astype(int)
    df["source_dataset"] = "Teen_Mental_Health_Dataset"
    return df


def load_dataset_2() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "Teen_Mental_Health_Dataset_2.csv")
    df = df.copy()

    risk_to_label = {"low": 0, "medium": 0, "high": 1}
    df["depression_label"] = df["depression_risk"].str.lower().map(risk_to_label)
    df = df.drop(columns=["depression_risk"])
    df["source_dataset"] = "Teen_Mental_Health_Dataset_2"
    return df


def main() -> None:
    df_1 = load_dataset_1()
    df_2 = load_dataset_2()

    common_columns = [
        "age",
        "gender",
        "daily_social_media_hours",
        "platform_usage",
        "sleep_hours",
        "screen_time_before_sleep",
        "academic_performance",
        "physical_activity",
        "social_interaction_level",
        "stress_level",
        "anxiety_level",
        "depression_label",
        "source_dataset",
    ]

    df_merged = pd.concat([df_1[common_columns], df_2[common_columns]], ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "dataset_merged.csv"
    df_merged.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(df_merged["depression_label"].value_counts().sort_index())
    print(df_merged["source_dataset"].value_counts())


if __name__ == "__main__":
    main()