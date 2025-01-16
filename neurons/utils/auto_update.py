import os
import subprocess
import git
import hashlib
import sys
import requests
from typing import Optional

from bittensor import logging

from .system import restart_app
import cli_parser

TARGET_BRANCH = "main"


class AutoUpdate:
    """
    Automatic update utility
    """

    def __init__(self):
        try:
            if not cli_parser.config.no_auto_update:
                self.repo = git.Repo(search_parent_directories=True)
                self.current_requirements_hash = self.get_requirements_hash()
        except Exception as e:
            logging.exception("Failed to initialize the repository", e)

    def get_local_latest_tag(self) -> Optional[git.Tag]:
        """
        Get the latest tag from the local git repository
        """
        try:
            tags = sorted(self.repo.tags, key=lambda t: t.commit.committed_datetime)
            current_tag: Optional[git.Tag] = tags[-1] if tags else None
            if current_tag:
                logging.info(f"Current tag: {current_tag.name}")
            return current_tag
        except Exception as e:
            logging.exception("Failed to get the current tag", e)
            return None

    def get_latest_release_tag(self):
        """
        Get the latest release tag from the GitHub repository
        """
        try:
            response = requests.get(
                f"https://api.github.com/repos/inference-labs-inc/omron-subnet/releases/latest"
            )
            response.raise_for_status()
            latest_release = response.json()
            return latest_release["tag_name"]
        except requests.RequestException as e:
            logging.exception("Failed to fetch the latest release from GitHub.", e)
            return None

    def get_requirements_hash(self):
        """
        Get the hash of the requirements.txt file
        """
        try:
            local_requirements_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "requirements.txt"
            )
            with open(local_requirements_path, "r", encoding="utf-8") as file:
                return hashlib.sha256(file.read().encode("utf-8")).hexdigest()
        except Exception as e:
            logging.exception("Failed to get the hash of the requirements file", e)
            return None

    def attempt_packages_update(self):
        """
        Attempt to update the packages by installing the requirements from the requirements.txt file
        """
        logging.info("Attempting to update packages...")

        try:
            repo = git.Repo(search_parent_directories=True)
            repo_path: str = (
                str(repo.working_tree_dir) if repo.working_tree_dir is not None else ""
            )

            requirements_path = os.path.join(repo_path, "requirements.txt")

            python_executable = sys.executable
            # trunk-ignore(bandit/B603)
            subprocess.check_call(
                [
                    python_executable,
                    "-m",
                    "ensurepip",
                ],
                timeout=60,
            )
            subprocess.check_call(
                [
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    requirements_path,
                    "-U",
                ],
                timeout=60,
            )
            logging.success("Successfully updated packages.")
        except Exception as e:
            logging.exception("Failed to update requirements", e)

    def update_to_latest_release(self) -> bool:
        """
        Update the repository to the latest release
        """
        try:

            if self.repo.is_dirty(untracked_files=False):
                logging.error(
                    "Current changeset is dirty. Please commit changes, discard changes or update manually."
                )
                return False

            latest_release_tag_name = self.get_latest_release_tag()
            if not latest_release_tag_name:
                logging.error("Failed to fetch the latest release tag.")
                return False

            current_tag = self.get_local_latest_tag()
            if current_tag.name == latest_release_tag_name:
                if self.repo.head.commit.hexsha == current_tag.commit.hexsha:
                    logging.info("Already having latest release locally.")
                    return True
                logging.info(
                    "Latest release is already checked out. But current commit is different."
                )
            else:
                logging.trace(
                    f"Attempting to check out the latest release: {latest_release_tag_name}..."
                )
                self.repo.remote().fetch(quiet=True, tags=True)
                if latest_release_tag_name not in [tag.name for tag in self.repo.tags]:
                    logging.error(
                        f"Latest release tag {latest_release_tag_name} not found in the repository."
                    )
                    return False

            self.repo.git.checkout(latest_release_tag_name)
            logging.success(
                f"Successfully checked out the latest release: {latest_release_tag_name}"
            )
            return True

        except Exception as e:
            logging.exception(
                "Automatic update failed. Manually pull the latest changes and update.",
                e,
            )

        return False

    def try_update(self):
        """
        Automatic update entrypoint method
        """

        if not self.update_to_latest_release():
            return

        if self.current_requirements_hash != self.get_requirements_hash():
            self.attempt_packages_update()

        logging.info("Restarting the application...")
        restart_app()
