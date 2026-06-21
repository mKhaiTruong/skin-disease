from abc import ABC, abstractmethod
import pandas as pd

class BaseTransformer(ABC):
    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """Transform source data → standard schema DataFrame"""
        pass