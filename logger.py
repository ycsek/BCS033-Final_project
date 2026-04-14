# logger.py
import json
import os
import logging
from datetime import datetime


class ExperimentLogger:
    def __init__(self, args):
        self.args = vars(args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "DP" if args.use_dp else "Base"
        self.log_dir = f"./logs/Static_{prefix}_{args.dataset}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(
                    self.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.results = {
            "hyperparameters": self.args,
            "trajectory": [],
            "final_asr": None
        }

    def info(self, msg):
        self.logger.info(msg)

    def log_epoch_metrics(self, epoch, target_loss, train_acc, target_acc, epsilon=None):
        record = {
            "epoch": epoch,
            "target_loss": target_loss,
            "train_acc": train_acc,      # 新增：训练集准确率
            "target_acc": target_acc     # 测试集准确率
        }
        if epsilon is not None:
            record["epsilon"] = epsilon
        self.results["trajectory"].append(record)

    def log_final_asr(self, asr):
        self.results["final_asr"] = asr

    def save_results(self):
        result_path = os.path.join(self.log_dir, "results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        self.info(
            f"\n>>> Experiment data successfully saved to {result_path} <<<")
