from core.constants import *
from core import read_yaml, create_directories

from training import TrainConfig, TrainMetrics

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_train_config(self) -> TrainConfig:
        config = self.config.train_config
        params = self.params.train_params
        create_directories([config.output_dir])

        return TrainConfig(
            train_dir   = Path(config.train_dir),
            valid_dir   = Path(config.valid_dir),
            output_dir  = Path(config.output_dir),
            resume_dir  = Path(config.resume_dir) if config.resume_dir else None,
            metrics     = TrainMetrics(
                model_name      = params.model_name,
                image_size      = params.image_size,
                epochs          = params.epochs,
                batch_size      = params.batch_size,
                lr              = params.lr,
                weight_decay    = params.weight_decay,
                unfreeze_epoch  = params.unfreeze_epoch,
                top_k_checkpoints = params.top_k_checkpoints,
                early_stopping  = params.early_stopping_patience
            )
        )