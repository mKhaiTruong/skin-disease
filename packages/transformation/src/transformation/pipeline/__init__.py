from transformation.config import ConfigurationManager
from transformation.components import Ingestion

class TransformationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        transform   = cfg_manager.get_transformation_config()
        transform   = Transformation(config=transform)
        transform.transform()