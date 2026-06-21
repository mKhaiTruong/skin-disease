import pandas as pd

from transformation.adapters.kaggle_adapter import KaggleTransformer

class TransformerFactory:
    _transformers = {
        "SKIN_DISEASES": KaggleTransformer,
    }
    
    @staticmethod
    def create(name: str, path: str) -> pd.DataFrame:
        cls = TransformerFactory._transformers.get(name)
        if not cls:
            raise ValueError(f"No transformer registered for: '{name}'. Add it to TransformerFactory._transformers.")
        return cls(src_path=path).transform()