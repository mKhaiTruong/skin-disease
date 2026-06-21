import os, sys
from core.logger import logger
from core.exception import CustomException
from ingestion import IngestionConfig
from ingestion.adapters.factory import IngestionAdapterFactory

class Ingestion:
    def __init__(self, config: IngestionConfig):
        self.config  = config
        self.adapter = IngestionAdapterFactory.create_adapter(
            self.config.src_type, self.config.source
        )
    
    def fetch_data(self) -> None:
        try: 
            dst = os.path.join(self.config.root_dir, self.config.name)
            self.adapter.fetch(dst)
            logger.info(f"Data fetched at {dst}")
        except Exception as e:
            raise CustomException(e, sys)