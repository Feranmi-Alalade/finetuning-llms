import json
import yaml
from pathlib import Path
from typing import Dict, Any


class RegressionGate:
    def __init__(self, contract_path: str):
        with open(contract_path, "r") as f:
            self.contract = yaml.safe_load(f)

    def check(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics satisfy the contract thresholds."""
        thresholds = self.contract.get("thresholds", {})
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] > threshold:
                    return False
        return True
