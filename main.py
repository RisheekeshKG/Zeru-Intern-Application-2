#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import os
from utils.user_data_extraction import CompoundExtractor 


def load_wallets(input_path):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")
    
    if "wallet_id" not in df.columns:
        raise ValueError("Input file must contain a 'wallet' column.")
    
    return df


def predict_scores(wallet_file, output_file, model_file="models/random_forest_model.joblib"):
    df_wallets = load_wallets(wallet_file)
    wallets = df_wallets["wallet_id"].dropna().unique().tolist()

    extractor = CompoundExtractor()
    features_df = extractor.extract_from_wallets(wallets)

    if "wallet" not in features_df.columns:
        raise ValueError("Extracted features must include a 'wallet' column.")

    X = features_df.drop(columns=["wallet"], errors="ignore")
    model = joblib.load(model_file)

    predictions = model.predict(X).clip(0, 1000)

    result_df = pd.DataFrame({
        "wallet_id": features_df["wallet"],
        "credit_score": predictions.round(2)
    })

    result_df.to_csv(output_file, index=False)
    print(f"✅ Credit scores saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Predict DeFi credit scores using a Random Forest model.")
    parser.add_argument("--input", type=str, default="data/compound_features_with_scores.csv", help="Path to input file (.csv or .xlsx) with wallet column")
    parser.add_argument("--output", type=str, default="outputs/scored_wallets_rf.csv", help="Output CSV file path")
    parser.add_argument("--model", type=str, default="models/random_forest_model.joblib", help="Trained model file (.joblib)")
    args = parser.parse_args()

    predict_scores(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
