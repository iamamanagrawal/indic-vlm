"""
Logging configuration module for the Indic VLM project.

This module sets up a centralized logger with both file and console handlers,
and provides decorators for function-level logging.
"""

import logging

logger = logging.getLogger("indic-vlm")
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create file handler
file_handler = logging.FileHandler(filename="logs/main.log", mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
