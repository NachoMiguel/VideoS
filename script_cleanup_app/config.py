#!/usr/bin/env python3
"""
Configuration for Script Cleanup App
"""

# Processing settings
MAX_SCRIPT_LENGTH = 100000  # Maximum script length to process
MIN_SCRIPT_LENGTH = 100     # Minimum script length to process

# Output settings
OUTPUT_DIR = "output"
INPUT_DIR = "input"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(levelname)s: %(message)s"

# Entity extraction settings
MAX_ENTITIES = 5  # Maximum entities to extract from script
MIN_ENTITY_LENGTH = 8  # Minimum entity name length 