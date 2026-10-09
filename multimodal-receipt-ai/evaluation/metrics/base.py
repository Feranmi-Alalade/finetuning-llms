from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """Compute metrics for a batch of predictions and references."""
        pass
