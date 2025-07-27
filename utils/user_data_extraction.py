import os
import requests
import pandas as pd
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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


class CompoundExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        subgraph_id: str = "4TbqVA8p2DoBd5qDbPMwmDZv3CsJjWtxo8nVSqF2tA9a",
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("Compound_V2_API")
        if not self.api_key:
            raise ValueError("API key not provided. Set env var `Compound_V2_API` or pass api_key=...")

        self.subgraph_id = subgraph_id
        self.base_url = base_url or f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def extract_from_wallets(self, wallets: List[str], show_progress: bool = True) -> pd.DataFrame:
        from tqdm import tqdm
        results = []
        iterator = tqdm(wallets, desc="Fetching wallets") if show_progress else wallets

        for w in iterator:
            try:
                acc = self._run_query(w)
                feat = self._extract_features(acc)
                if feat:
                    results.append(feat)
            except Exception as e:
                print(f"[!] Failed for {w}: {e}")
        return pd.DataFrame(results)

    def extract_from_excel(self, path: str, column: str = "wallet_id") -> pd.DataFrame:
        wallets = pd.read_excel(path)[column].dropna().astype(str).unique().tolist()
        return self.extract_from_wallets(wallets)

    def extract_from_csv(self, path: str, column: str = "wallet_id") -> pd.DataFrame:
        wallets = pd.read_csv(path)[column].dropna().astype(str).unique().tolist()
        return self.extract_from_wallets(wallets)

    def _run_query(self, wallet: str) -> Optional[Dict]:
        payload = {"query": GRAPHQL_QUERY, "variables": {"addr": wallet.lower()}}
        r = requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return data["data"]["account"]

    @staticmethod
    def _sum_usd(events: Optional[List[Dict]]) -> float:
        if not events:
            return 0.0
        return sum(float(e.get("amountUSD", 0) or 0) for e in events)

    @staticmethod
    def _active_days(*event_lists: List[Dict]) -> int:
        ts = []
        for lst in event_lists:
            if lst:
                ts.extend(int(e.get("timestamp", 0) or 0) for e in lst if "timestamp" in e)
        if not ts:
            return 0
        return max((max(ts) - min(ts)) // 86400, 0)

    def _extract_features(self, account: Optional[Dict]) -> Dict:
        if not account:
            return {}

        borrows = account.get("borrows", [])
        deposits = account.get("deposits", [])
        repays = account.get("repays", [])
        liquidates = account.get("liquidates", [])
        liquidations = account.get("liquidations", [])

        return {
            "wallet": account["id"],
            "borrow_count": int(account.get("borrowCount", 0)),
            "deposit_count": int(account.get("depositCount", 0)),
            "repay_count": int(account.get("repayCount", 0)),
            "liquidation_count": int(account.get("liquidationCount", 0)),
            "liquidates_count": int(account.get("liquidateCount", 0)),
            "withdraw_count": int(account.get("withdrawCount", 0)),

            "total_borrow_usd": self._sum_usd(borrows),
            "total_repay_usd": self._sum_usd(repays),
            "total_deposit_usd": self._sum_usd(deposits),
            "total_liquidated_usd": self._sum_usd(liquidates),
            "total_got_liquidated_usd": self._sum_usd(liquidations),

            "active_days": self._active_days(borrows, repays, deposits)
        }
