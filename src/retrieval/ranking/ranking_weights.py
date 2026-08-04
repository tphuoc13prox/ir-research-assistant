import yaml
from pathlib import Path

DEFAULT_WEIGHTS = {
    "title": 4.0,
    "keywords": 3.0,
    "abstract": 2.0,
    "body": 1.0,
    "phrase": 2.0,
}

class RankingConfig:
    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.enabled = True
        self.rrf_weight = 0.7
        self.field_weight = 0.3
        self.weights = DEFAULT_WEIGHTS.copy()
        self.load()

    def load(self) -> None:
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                retrieval_data = data.get("retrieval", {})
                ranking_data = retrieval_data.get("ranking", {})
                
                self.enabled = ranking_data.get("enabled", True)
                self.rrf_weight = float(ranking_data.get("rrf_weight", 0.7))
                self.field_weight = float(ranking_data.get("field_weight", 0.3))
                
                weights_data = ranking_data.get("weights", {})
                for field, val in DEFAULT_WEIGHTS.items():
                    self.weights[field] = float(weights_data.get(field, val))
        except Exception as exc:
            print(f"Warning: Failed to load ranking configuration ({exc}). Using defaults.")
            self.enabled = True
            self.rrf_weight = 0.7
            self.field_weight = 0.3
            self.weights = DEFAULT_WEIGHTS.copy()
