import os
import shutil
import atexit
import signal
from constants import TEMP_FOLDER


def cleanup_temp_dir(signum=None, frame=None):
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)


def register_cleanup_handlers():
    atexit.register(cleanup_temp_dir)
    signal.signal(signal.SIGTERM, cleanup_temp_dir)
    signal.signal(signal.SIGINT, cleanup_temp_dir)
