import logging
# src/utils/helpers.py
import os
import yaml

def load_config(path="config/config.yaml"):
    with open(path, 'r') as f:
        content = f.read()

    # Replace ${VAR_NAME} with value from environment
    for key, value in os.environ.items():
        content = content.replace(f"${{{key}}}", value)

    return yaml.safe_load(content)


def setup_logger(name="litho"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
