import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("Compound_V2_API")

SUBGRAPH_URL = "https://gateway.thegraph.com/api/subgraphs/id/4TbqVA8p2DoBd5qDbPMwmDZv3CsJjWtxo8nVSqF2tA9a"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

GRAPHQL_QUERY = """
query($addr: ID!) {
  account(id: $addr) {
    id
    borrowCount
    depositCount
    liquidateCount
    liquidationCount
    repayCount
    withdrawCount
    borrows { amountUSD timestamp }
    deposits { amountUSD timestamp }
    repays { amountUSD timestamp }
    liquidates { amountUSD timestamp }
    liquidations { amountUSD timestamp }
  }
}
"""

def run_query(wallet):
    response = requests.post(
        SUBGRAPH_URL,
        headers=HEADERS,
        json={"query": GRAPHQL_QUERY, "variables": {"addr": wallet.lower()}},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]["account"]

def extract_features(account):
    if not account:
        return {}

    def total_usd(events):
        return sum(float(e.get("amountUSD", 0)) for e in events)

    def active_days(events):
        timestamps = [int(e.get("timestamp", 0)) for e in events if "timestamp" in e]
        if not timestamps:
            return 0
        days = (max(timestamps) - min(timestamps)) // 86400
        return max(days, 0)

    return {
        "wallet": account["id"],
        "borrow_count": int(account.get("borrowCount", 0)),
        "deposit_count": int(account.get("depositCount", 0)),
        "repay_count": int(account.get("repayCount", 0)),
        "liquidation_count": int(account.get("liquidationCount", 0)),
        "liquidates_count": int(account.get("liquidateCount", 0)),
        "withdraw_count": int(account.get("withdrawCount", 0)),

        "total_borrow_usd": total_usd(account.get("borrows", [])),
        "total_repay_usd": total_usd(account.get("repays", [])),
        "total_deposit_usd": total_usd(account.get("deposits", [])),
        "total_liquidated_usd": total_usd(account.get("liquidates", [])),
        "total_got_liquidated_usd": total_usd(account.get("liquidations", [])),

        "active_days": active_days(account.get("borrows", []) +
                                   account.get("repays", []) +
                                   account.get("deposits", []))
    }

def main():
    
    df = pd.read_excel("./data/wallets.xlsx") 
    wallets = df["wallet_id"].dropna().unique().tolist()

    all_features = []
    for wallet in tqdm(wallets, desc="Processing wallets"):
        try:
            account_data = run_query(wallet)
            features = extract_features(account_data)
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"[!] Failed for {wallet}: {e}")

    features_df = pd.DataFrame(all_features)
    features_df.to_csv("./data/compound_features.csv", index=False)
    print("Saved compound_features.csv")

if __name__ == "__main__":
    main()