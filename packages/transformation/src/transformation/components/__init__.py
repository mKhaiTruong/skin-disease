import os, sys, numpy as np
from PIL import Image
import albumentations as A

from core.logger import logger
from core.exception import CustomException

from transformation import TransformationConfig
from transformation.adapters.factory import TransformerFactory

class Transformation:
    def __init__(self, config: TransformationConfig):
        self.config   = config
        self.dfs = [
            (TransformerFactory.create(d["name"], d["path"]), d)
            for d in self.config.data_dirs
            if d["name"] in TransformerFactory._transformers
        ]
        logger.info(f"Loaded {len(self.dfs)} transformer(s): {[d['name'] for _, d in self.dfs]}")
    
    def _prepare_output_dirs(self):
        for d in [self.config.out_train_dir, self.config.out_valid_dir, self.config.out_infer_dir]:
            os.makedirs(d, exist_ok=True)
            
    def _get_resizer(self):
        size = self.config.metrics.image_size
        return A.Compose([A.Resize(size, size)])
    
    def _save_split(self, df_split, out_dir, resizer):
        for _, row in df_split.iterrows():
            label_dir = out_dir / row["label"]
            os.makedirs(label_dir, exist_ok=True)
            
            img = Image.open(row["image_path"]).convert("RGB")
            resized = resizer(image=np.array(img))["image"]

            out_path = label_dir / Path(row["image_path"]).name
            Image.fromarray(resized).save(out_path)
    
    def transform(self):
        try:
            self._prepare_output_dirs()
            metrics = self.config.metrics
            resizer = self._get_resizer()
            
            for df, d in self.dfs:
                name = d["name"].lower()
                df   = df.sample(frac=1, random_state=42).reset_index(drop=True)
                n    = len(df)
                
                train_end = int(n * metrics.train_ratio)
                valid_end = int(n * (metrics.train_ratio + metrics.valid_ratio))
                
                train_df = df.iloc[:train_end]
                valid_df = df.iloc[train_end:valid_end]
                infer_df = df.iloc[valid_end:]
                
                logger.info(f"{name}: train={len(train_df)} | valid={len(valid_df)} | infer={len(infer_df)}")
                
                self._save_split(train_df, self.config.out_train_dir, resizer)
                self._save_split(valid_df, self.config.out_valid_dir, resizer)
                self._save_split(infer_df, self.config.out_infer_dir, resizer)
            
        except Exception as e:
            raise CustomException(e, sys)