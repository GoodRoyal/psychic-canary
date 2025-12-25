from pydantic import BaseModel
from typing import List, Dict

class InferenceRequest(BaseModel):
    tickers: List[str] = ["SPY", "AGG"]  # default assets
    days: int = 756  # ~3 years data

class InferenceResponse(BaseModel):
    tickers: List[str]
    phase: str  # "exploitation", "exploration", "crisis_imminent"
    warning: bool
    temperature: float
    entropy: float
    free_energy: float
    curvature_proxy: float
    vol_ratio: float
    optimal_weights: Dict[str, float]
    suggested_action: str
    holonomic_note: str
    data_points: int
