"""
Utility functions for liver segmentation project.
"""
import os
import logging

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directories(base_path):
    """
    Create necessary directories.
    Args:
        base_path (str): Base directory path
    """
    dirs = ['train', 'val', 'test', 'models']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)
