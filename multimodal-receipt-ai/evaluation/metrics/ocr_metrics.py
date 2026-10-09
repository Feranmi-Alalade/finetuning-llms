from typing import List, Dict
from multimodal_receipt_ai.evaluation.metrics.base import BaseMetric
import jiwer


class OCRMetric(BaseMetric):
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        cer = jiwer.cer(references, predictions)
        wer = jiwer.wer(references, predictions)
        return {"cer": cer, "wer": wer}
