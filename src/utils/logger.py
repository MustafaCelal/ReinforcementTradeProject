import os
import json
import logging
from datetime import datetime
import config.settings as cfg

class TradeLogger:
    def __init__(self, log_dir=None, filename="trade_log.jsonl"):
        if log_dir is None:
            log_dir = os.path.join(cfg.OUTPUTS_DIR, "logs")
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        
        # Configure standard logging for console/file
        self.logger = logging.getLogger("TradeLogger")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # File handler (text)
            fh = logging.FileHandler(os.path.join(log_dir, "trading.log"))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log_step(self, step_data):
        """Append step data as a JSON line"""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(step_data) + "\n")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
