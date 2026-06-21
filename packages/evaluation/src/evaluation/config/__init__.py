from core.constants import *
from core import read_yaml, create_directories

from evaluation import EvaluationConfig, EvaluationMetrics

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_eval_config(self) -> EvaluationConfig:
        config = self.config.eval_config
        params = self.params.eval_params
        
        output_dir = Path(config.output_dir) / params.model_name
        create_directories([config.output_dir, output_dir])

        return EvaluationConfig(
            train_dir   = Path(config.train_dir),
            output_dir  = output_dir,
            data_dir    = Path(config.data_dir),
            metrics     = EvaluationMetrics(
                model_name  = params.model_name,
                batch_size  = params.batch_size,
            )
        )