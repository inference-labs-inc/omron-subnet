import hashlib
import os
import shutil
import subprocess
import sys
from typing import Optional

import bittensor as bt
import git
import torch
import wandb_logger
import requests
import json
from __init__ import __version__

TARGET_BRANCH = "main"


def sync_model_files():
    """
    Sync external model files
    """
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "deployment_layer")
    SYNC_LOG_PREFIX = "  SYNC  | "

    for model_hash in os.listdir(MODEL_DIR):
        if not model_hash.startswith("model_") or model_hash == "model_0":
            continue

        metadata_file = os.path.join(MODEL_DIR, model_hash, "metadata.json")
        if not os.path.isfile(metadata_file):
            bt.logging.error(
                SYNC_LOG_PREFIX
                + f"Metadata file not found at {metadata_file} for {model_hash}. Skipping sync for this model."
            )
            continue

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            bt.logging.error(
                SYNC_LOG_PREFIX + f"Failed to parse JSON from {metadata_file}"
            )
            continue

        external_files = metadata.get("external_files", {})
        for key, url in external_files.items():
            file_path = os.path.join(MODEL_DIR, model_hash, key)
            if os.path.isfile(file_path):
                bt.logging.info(
                    SYNC_LOG_PREFIX
                    + f"File {key} for {model_hash} already downloaded, skipping..."
                )
                continue

            bt.logging.info(SYNC_LOG_PREFIX + f"Downloading {url} to {file_path}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except requests.RequestException as e:
                bt.logging.error(
                    SYNC_LOG_PREFIX + f"Failed to download {url} to {file_path}: {e}"
                )
                continue


def hotkey_to_split_tensor(hotkey_ss58=None):
    """
    Converts a hotkey into a tensor of ascii numbers representing the hotkey SS58
    """
    if hotkey_ss58 is None:
        bt.logging.error("No hotkey provided, cannot split into a tensor")
        return None

    split = [ord(c) for c in hotkey_ss58]
    return torch.tensor(
        split,
        dtype=torch.float,
    )


def split_tensor_to_hotkey(split_tensor):
    return "".join([chr(i) for i in split_tensor.tolist()])


def restart_app():
    """
    Restart the application to apply the updated changes
    """
    bt.logging.success("App restarting due to auto-update")
    python = sys.executable
    os.execl(python, python, *sys.argv)


def clean_temp_files():
    """
    Clean temporary files
    """
    bt.logging.info("Deleting temp folder...")
    folder_path = os.path.join(
        os.path.dirname(__file__),
        "execution_layer",
        "temp",
    )
    if os.path.exists(folder_path):
        bt.logging.debug("Removing temp folder...")
        shutil.rmtree(folder_path)


class AutoUpdate:
    """
    Automatic update utility
    """

    def __init__(self):
        try:
            self.repo = git.Repo(search_parent_directories=True)
            self.update_requirements = False
        except Exception as e:
            bt.logging.exception("Failed to initialize the repository", e)

    def convert_version_str_to_int(self, version):
        """
        Converts the provided version string in the format 0.0.0 into an integer
        """
        return int(version.replace(".", ""))

    def get_remote_status(self) -> Optional[str]:
        """
        Fetch the remote version string from the neurons/__init__.py file in the current repository
        """
        try:
            # Perform a git fetch to ensure we have the latest remote information
            self.repo.remotes.origin.fetch(kill_after_timeout=5)

            # Check if the requirements.txt file has changed
            local_requirements_path = os.path.join(
                os.path.dirname(__file__), "..", "requirements.txt"
            )
            with open(local_requirements_path, "r", encoding="utf-8") as file:
                local_requirements_hash = hashlib.sha256(
                    file.read().encode("utf-8")
                ).hexdigest()
            requirements_blob = (
                self.repo.remote().refs[TARGET_BRANCH].commit.tree / "requirements.txt"
            )
            remote_requirements_content = requirements_blob.data_stream.read().decode(
                "utf-8"
            )
            remote_requirements_hash = hashlib.sha256(
                remote_requirements_content.encode("utf-8")
            ).hexdigest()
            self.update_requirements = (
                local_requirements_hash != remote_requirements_hash
            )

            # Get version number from remote
            blob = (
                self.repo.remote().refs[TARGET_BRANCH].commit.tree
                / "neurons"
                / "__init__.py"
            )
            lines = blob.data_stream.read().decode("utf-8").split("\n")

            for line in lines:
                if line.startswith("__version__"):
                    version_info = line.split("=")[1].strip(" \"'").replace('"', "")
                    return version_info
        except Exception as e:
            bt.logging.exception(
                "Failed to get remote file content for version check", e
            )
            return None

    def check_version_updated(self):
        """
        Compares local and remote versions and returns True if the remote version is higher
        """
        remote_version = self.get_remote_status()
        if not remote_version:
            bt.logging.error("Failed to get remote version, skipping version check")
            return False

        local_version = __version__

        local_version_int = self.convert_version_str_to_int(local_version)
        remote_version_int = self.convert_version_str_to_int(remote_version)
        wandb_logger.safe_log(
            {"local_version": local_version_int, "remote_version": remote_version_int}
        )

        bt.logging.info(
            f"Version check - remote_version: {remote_version}, local_version: {local_version}"
        )

        if remote_version_int > local_version_int:
            bt.logging.info(
                f"Remote version ({remote_version}) is higher "
                f"than local version ({local_version}), automatically updating..."
            )
            return True
        return False

    def attempt_update(self):
        """
        Attempt to update the repository by pulling the latest changes from the remote repository
        """
        try:

            origin = self.repo.remotes.origin

            if self.repo.is_dirty(untracked_files=False):
                bt.logging.error(
                    "Current changeset is dirty. Please commit changes, discard changes or update manually."
                )
                return False
            try:
                bt.logging.trace("Attempting to pull latest changes...")
                origin.pull(kill_after_timeout=10)
                bt.logging.success("Successfully pulled the latest changes")
                return True
            except git.exc.GitCommandError as e:
                bt.logging.exception(
                    "Automatic update failed due to conflicts. Attempting to handle merge conflicts...",
                    e,
                )
                return self.handle_merge_conflicts()

        except Exception as e:
            bt.logging.exception(
                "Automatic update failed. Manually pull the latest changes and update.",
                e,
            )

        return False

    def handle_merge_conflicts(self):
        """
        Attempt to automatically resolve any merge conflicts that may have arisen
        """
        try:
            self.repo.git.reset("--merge")
            origin = self.repo.remotes.origin
            current_branch = self.repo.active_branch
            origin.pull(current_branch.name)

            for item in self.repo.index.diff(None):
                file_path = item.a_path
                bt.logging.info(f"Resolving conflict in file: {file_path}")
                self.repo.git.checkout("--theirs", file_path)
            self.repo.index.commit("Resolved merge conflicts automatically")
            bt.logging.info(
                "Merge conflicts resolved, repository updated to remote state."
            )
            bt.logging.info("✅ Successfully updated")
            return True
        except git.GitCommandError as e:
            bt.logging.exception(
                "Failed to resolve merge conflicts, automatic update cannot proceed. Please manually pull and update.",
                e,
            )
            return False

    def attempt_package_update(self):
        """
        Attempt to update the packages by installing the requirements from the requirements.txt file
        """
        bt.logging.info("Attempting to update packages...")

        try:
            repo = git.Repo(search_parent_directories=True)
            repo_path = repo.working_tree_dir

            requirements_path = os.path.join(repo_path, "requirements.txt")

            python_executable = sys.executable
            subprocess.check_call(
                [python_executable, "-m", "pip", "install", "-r", requirements_path],
                timeout=60,
            )
            bt.logging.success("Successfully updated packages.")
        except Exception as e:
            bt.logging.exception("Failed to update requirements", e)

    def try_update(self):
        """
        Automatic update entrypoint method
        """
        if self.repo.head.is_detached or self.repo.active_branch.name != "main":
            bt.logging.warning("Not on the main branch, skipping auto-update")
            return

        if not self.check_version_updated():
            return

        if not self.attempt_update():
            return

        if self.update_requirements:
            self.attempt_package_update()

        restart_app()
