from training.config import ConfigurationManager
from training.components import Train

class IngestionPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        train_cfg   = cfg_manager.get_train_config()
        train_cls   = Train(config=train_cfg)
        train_cls.run()