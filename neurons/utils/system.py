import os
import sys
import shutil
import functools
import multiprocessing
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


def timeout_with_multiprocess(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target_func(result_dict, *args, **kwargs):
                try:
                    result_dict["result"] = func(*args, **kwargs)
                except Exception as e:
                    result_dict["exception"] = e

            manager = multiprocessing.Manager()
            result_dict = manager.dict()
            process = multiprocessing.Process(
                target=target_func, args=(result_dict, *args), kwargs=kwargs
            )
            process.start()
            process.join(seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                logging.warning(
                    f"Function '{func.__name__}' timed out after {seconds} seconds"
                )
                return None

            if "exception" in result_dict:
                raise result_dict["exception"]

            return result_dict.get("result", None)

        return wrapper

    return decorator
