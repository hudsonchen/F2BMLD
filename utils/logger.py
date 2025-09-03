import logging
import json
import os

import os
import logging

class StandardLogger:
    def __init__(self, name='learner', log_dir='./logs'):
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'{name}.log')

        # 🔑 Remove old log file if it exists
        if os.path.exists(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear old handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def write(self, result_dict):
        msg = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
             for k, v in result_dict.items()]
        )
        self.logger.info(msg)
