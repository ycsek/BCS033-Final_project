# logger.py
import json
import os
import logging
from datetime import datetime


class ExperimentLogger:
    def __init__(self, args):
        self.args = vars(args)
        # 以当前时间戳创建独立的实验日志文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "DP" if args.use_dp else "Base"
        self.log_dir = f"./logs/{prefix}_{args.dataset}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        # 配置 Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(
                    self.log_dir, "training.log")),
                logging.StreamHandler()  # 同时输出到终端
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 初始化需要保存的 JSON 数据结构
        self.results = {
            "hyperparameters": self.args,
            "target_model_metrics": [],
            "mia_metrics": []
        }

    def info(self, msg):
        self.logger.info(msg)

    def log_target_epoch(self, epoch, loss, acc, epsilon=None):
        record = {"epoch": epoch, "loss": loss, "acc": acc}
        if epsilon is not None:
            record["epsilon"] = epsilon
        self.results["target_model_metrics"].append(record)

    def log_mia_epoch(self, epoch, loss, asr):
        self.results["mia_metrics"].append(
            {"epoch": epoch, "loss": loss, "asr": asr})

    def save_results(self):
        result_path = os.path.join(self.log_dir, "results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        self.info(f"Experiment data successfully saved to {result_path}")
