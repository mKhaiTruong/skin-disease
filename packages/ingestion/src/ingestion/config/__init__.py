import os
from itertools import chain
from typing import List

from dotenv import load_dotenv
load_dotenv()

from core.constants import *
from core import read_yaml, create_directories
from ingestion import IngestionConfig

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def _parse_sources(self, env_key: str) -> list:
        raw = os.getenv(env_key, "")
        if not raw.strip():
            return []
        
        sources = []
        for entry in raw.split(","):
            parts = entry.strip().split(":")
            
            if len(parts) != 3:
                continue
            
            name, src_type, source = parts
            sources.append({
                "name": name, 
                "src_type": src_type, 
                "source": source
            })
        
        return sources

    def get_ingestion_configs(self) -> List[IngestionConfig]:
        config = self.config.ingestion_config
        create_directories([config.root_dir])

        # ----- DEFINING ALL SOURCES <<<<<<<<<<<<
        local_src  = self._parse_sources("LOCAL_SOURCES")
        kaggle_src = self._parse_sources("KAGGLE_SOURCES")
        ee_src     = self._parse_sources("EE_SOURCES")
        
        configs = []
        for s in chain(local_src, kaggle_src, ee_src):
            configs.append(IngestionConfig(
                root_dir     = config.root_dir,
                src_type     = s["src_type"],
                source       = s["source"],
                name         = s["name"],
            ))

        return configs