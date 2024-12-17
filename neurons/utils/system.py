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


def timeout_with_multiprocess_retry(seconds, retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                logging.info(f"Attempt {attempt + 1} of {retries}")

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
                    if attempt < retries - 1:
                        logging.info(f"Retrying... ({attempt + 1}/{retries})")
                        continue
                    return None

                if "exception" in result_dict:
                    if attempt < retries - 1:
                        logging.info(
                            f"Retrying due to exception... ({attempt + 1}/{retries})"
                        )
                        continue
                    raise result_dict["exception"]

                result = result_dict.get("result", None)
                if result:
                    return result
                elif attempt < retries - 1:
                    logging.info(
                        f"Retrying due to falsy result... ({attempt + 1}/{retries})"
                    )
                    continue
                if func.__name__ == "update_weights":
                    logging.error(
                        f"Failed to set weights after {retries} attempts. "
                        "Another attempt will be made after the next request cycle."
                    )
                else:
                    logging.error(
                        f"Function '{func.__name__}' returned {result} after {retries} attempts"
                    )
                return None

            return None

        return wrapper

    return decorator
