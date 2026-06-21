from ingestion.config import ConfigurationManager
from ingestion.components import Ingestion

class IngestionPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager     = ConfigurationManager()
        ingestion_cfgs  = cfg_manager.get_ingestion_configs()
        for cfg in ingestion_cfgs:
            Ingestion(config=cfg).fetch_data()