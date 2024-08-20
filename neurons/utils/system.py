import os
import sys
import shutil
from bittensor import logging


def restart_app():
    """
    Restart the application to apply the updated changes
    """
    logging.success("App restarting due to auto-update")
    python = sys.executable
    # trunk-ignore(bandit/B606)
    os.execl(python, python, *sys.argv)


def clean_temp_files():
    """
    Clean temporary files
    """
    logging.info("Deleting temp folder...")
    folder_path = os.path.join(
        os.path.dirname(__file__),
        "execution_layer",
        "temp",
    )
    if os.path.exists(folder_path):
        logging.debug("Removing temp folder...")
        shutil.rmtree(folder_path)
