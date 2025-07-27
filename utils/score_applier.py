import pandas as pd

INPUT_FILE  = "./data/compound_features.csv"
OUTPUT_FILE = "./data/compound_features_with_scores.csv"


def compute_score(row):
    score = 500  # 500 is good for average i guess

    if row["total_borrow_usd"] > 0:
        repay_ratio = min(row["total_repay_usd"] / row["total_borrow_usd"], 2)
        score += 150 * repay_ratio  

    # Deposit-to-Borrow Ratio (0–100)
    if row["total_borrow_usd"] > 0:
        deposit_ratio = min(row["total_deposit_usd"] / row["total_borrow_usd"], 2)
        score += 100 * deposit_ratio

    #Liquidation USD penalty (0 to -150)
    if row["total_borrow_usd"] > 0:
        liq_usd_penalty = min(row["total_got_liquidated_usd"] / row["total_borrow_usd"], 1.0)
        score -= 150 * liq_usd_penalty

    # Number of Liquidations (0 to -100)
    score -= min(row["liquidation_count"], 5) * 20

    # Withdraw penalty (0 to -50)
    score -= min(row["withdraw_count"], 5) * 10

    # Liquidator reward (0 to +50)
    score += min(row["total_liquidated_usd"] / 10_000, 1.0) * 50

    # Active duration (0 to +50)
    score += min(row["active_days"] / 365, 1.0) * 50

    # Full repayment bonus (+50)
    if row["borrow_count"] > 0 and row["repay_count"] >= row["borrow_count"]:
        score += 50

    return int(min(1000, max(0, score)))




def main():
    df = pd.read_csv(INPUT_FILE)
    df["score"] = df.apply(compute_score, axis=1)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Scores saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
