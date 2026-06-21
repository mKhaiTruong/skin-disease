import sys
import torch

from core.logger import logger
from core.exception import CustomException

from vision.strategies.onnx_factory import ModelFactory
from training import TrainConfig
from training.utils.loaders import build_dataloaders
from training.components.wrapper import Trainer
from training.utils.engine import save_model_metadata, push_to_s3

class Train:
    def __init__(self, config: TrainConfig):
        self.config = config

    def run(self):
        try:
            train_loader, val_loader, num_classes, class_counts = build_dataloaders(
                self.config.train_dir, self.config.valid_dir, self.config.metrics.batch_size
            )
            logger.info(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}  Classes: {num_classes}")

            model = ModelFactory.create(self.config.metrics.model_name, self.config, num_classes)
            trainer = Trainer(model, class_counts, self.config)
            trainer.fit(train_loader, val_loader)

            best_checkpoint = trainer.checkpoint_mgr.get_best_checkpoint()
            trainer.load_checkpoint(best_checkpoint)

            sample_input = torch.randn(1, 3, self.config.metrics.image_size, self.config.metrics.image_size).to(model.device)
            onnx_path = model.checkpoint_dir / "weights.onnx"
            model.save_onnx(onnx_path, sample_input)

            save_model_metadata(
                output_dir=model.checkpoint_dir,
                class_names=train_loader.dataset.classes,
                image_size=self.config.metrics.image_size
            )

            push_to_s3(model.checkpoint_dir, prefix=f"train/{self.config.metrics.model_name}")

            logger.info(f"Training complete. Best val_acc={trainer.best_val_acc:.4f}")

        except Exception as e:
            raise CustomException(e, sys)