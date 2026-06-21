from evaluation.config import ConfigurationManager
from evaluation.components import Evaluation

class EvaluationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        eval_cfg    = cfg_manager.get_eval_config()
        evaluation  = Evaluation(config=eval_cfg)
        evaluation.evaluate()