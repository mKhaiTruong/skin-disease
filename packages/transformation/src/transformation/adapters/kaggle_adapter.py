from pathlib import Path
import pandas as pd
from core.logger import logger

from transformation.adapters import BaseTransformer

class KaggleTransformer(BaseTransformer):
    def __init__(self, src_path: str):
        self.src_path = src_path
    
    def transform(self) -> pd.DataFrame:
        actual_path = self.src_path
        while True:
            subdirs = [d for d in actual_path.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                actual_path = subdirs[0]  # going through folders until reaching data
            else:
                break
        
        records = []
        for class_dir in actual_path.iterdir():
            if not class_dir.is_dir():
                continue
        
            label = class_dir.name
            for img_path in class_dir.glob("*.*"):
                records.append({"image_path": str(img_path), "label": label})

        df = pd.DataFrame(records)
        logger.info(f"KaggleTransformer: loaded {len(df)} records from {actual_path}")
        return df