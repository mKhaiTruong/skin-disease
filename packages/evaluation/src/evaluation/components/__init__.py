import sys
from pathlib import Path
from core.logger import logger
from core.exception import CustomException

from vision.strategies.onnx_factory import EvalModelFactory
from evaluation import EvaluationConfig
from evaluation.utils.loader import build_eval_loaders
from evaluation.utils.metrics import collect_predictions, compute_full_metrics

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        try:
            eval_loader, classes = build_eval_loaders(self.config.data_dir, self.config.metrics.batch_size)
            logger.info(f"Evaluation dataset: {len(eval_loader.dataset):,} Classes: {classes}")

            model = EvalModelFactory.create(self.config.metrics.model_name, len(classes))
            best_checkpoint = self._find_best_checkpoint()
            model.load(best_checkpoint)

            y_true, y_pred = collect_predictions(model, eval_loader)
            metrics = compute_full_metrics(y_true, y_pred, classes)

            logger.info(f"Eval accuracy: {metrics['report']['accuracy']:.4f}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    def _find_best_checkpoint(self) -> Path:
        ckpt_dir = self.config.train_dir / self.config.metrics.model_name
        checkpoints = list(ckpt_dir.glob("epoch*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        return max(checkpoints, key=lambda p: float(p.stem.split("_acc")[1]))