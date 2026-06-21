import os, sys, logging
from pathlib import Path

logger = logging.getLogger("skin_disease_core")
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


log_dir = ROOT_DIR / "logs"
log_filepath = log_dir / "running_logs.log"
os.makedirs(log_dir, exist_ok=True)


formatter = logging.Formatter(LOG_FORMAT)
handler_file = logging.FileHandler(log_filepath)
handler_console = logging.StreamHandler(sys.stdout)

handler_file.setFormatter(formatter)
handler_console.setFormatter(formatter)

logger.setLevel(logging.INFO)
logger.addHandler(handler_file)
logger.addHandler(handler_console)