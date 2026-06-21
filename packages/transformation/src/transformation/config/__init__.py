from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from core.constants import *
from core import read_yaml, create_directories

from transformation import TransformationConfig, TransformationMetrics
from transformation.utils import _build_data_dirs, _parse_env_sources

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_transformation_config(self) -> TransformationConfig:
        config = self.config.transformation_config
        params = self.params.transformation_params
        create_directories([config.root_dir])

        data_dirs = _build_data_dirs(
            Path(self.config.ingestion_config.root_dir),
            _parse_env_sources("LOCAL_SOURCES"),
            _parse_env_sources("KAGGLE_SOURCES"),
        )

        return TransformationConfig(
            root_dir        = Path(config.root_dir),
            data_dirs       = data_dirs,
            out_train_dir   = Path(config.root_dir) / "train",
            out_valid_dir   = Path(config.root_dir) / "valid",
            out_infer_dir   = Path(config.root_dir) / "infer",
            metrics = TransformationMetrics(
                image_size  = params.image_size,
                train_ratio = params.train_ratio,
                valid_ratio = params.valid_ratio
            ),
        )