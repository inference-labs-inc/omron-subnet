import os
import shutil
import atexit
import signal
from utils.system import get_temp_folder
import bittensor as bt


def cleanup_temp_dir(signum=None, frame=None):
    temp_folder = get_temp_folder()
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                bt.logging.error(f"Error cleaning up temp directory: {e}")
                pass


def register_cleanup_handlers():
    atexit.register(cleanup_temp_dir)
    signal.signal(signal.SIGTERM, cleanup_temp_dir)
    signal.signal(signal.SIGINT, cleanup_temp_dir)
