"""
The MIT License (MIT)
Copyright © 2023 Chris Wilson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import os
import subprocess
import sys

import bittensor as bt
import git
import requests

from __init__ import __version__


def version2number(version):
    return int(version.replace(".", "").replace("-", "").replace("_", ""))


def get_remote_version():
    url = "https://raw.githubusercontent.com/inference-labs-inc/omron-subnet/main/neurons/__init__.py"
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.split("\n")
        for line in lines:
            if line.startswith("__version__"):
                version_info = line.split("=")[1].strip(" \"'").replace('"', "")
                return version_info
    else:
        bt.logging.error("Failed to get remote file content for version check")


def get_local_version():
    return __version__


def check_version_updated():
    remote_version = get_remote_version()
    local_version = get_local_version()
    bt.logging.info(
        f"Version check - remote_version: {remote_version}, local_version: {local_version}"
    )

    if version2number(remote_version) > version2number(local_version):
        bt.logging.info(
            f"Remote version ({remote_version}) is higher than local version ({local_version}), automatically updating..."
        )
        return True
    else:
        return False


def update_repo():
    try:
        repo = git.Repo(search_parent_directories=True)

        origin = repo.remotes.origin

        if repo.is_dirty(untracked_files=True):
            bt.logging.error(
                "Update failed: Uncommited changes detected. Please commit changes"
            )
            return False
        try:
            bt.logging.info("Try pulling remote repository")
            origin.pull()
            bt.logging.info("pulling success")
            return True
        except git.exc.GitCommandError as e:
            bt.logging.info(
                f"update : Merge conflict detected: {e} Recommend you manually commit changes and update"
            )
            return handle_merge_conflict(repo)

    except Exception as e:
        bt.logging.error(
            f"update failed: {e} Recommend you manually commit changes and update"
        )

    return False


def handle_merge_conflict(repo):
    try:
        repo.git.reset("--merge")
        origin = repo.remotes.origin
        current_branch = repo.active_branch
        origin.pull(current_branch.name)

        for item in repo.index.diff(None):
            file_path = item.a_path
            bt.logging.info(f"Resolving conflict in file: {file_path}")
            repo.git.checkout("--theirs", file_path)
        repo.index.commit("Resolved merge conflicts automatically")
        bt.logging.info("Merge conflicts resolved, repository updated to remote state.")
        bt.logging.info("✅ Successfully updated")
        return True
    except git.GitCommandError as e:
        bt.logging.error(
            f"update failed: {e} Recommend you manually commit changes and update"
        )
        return False


def version2number(version_string):
    version_digits = version_string.split(".")
    return 100 * version_digits[0] + 10 * version_digits[1] + version_digits[2]


def restart_app():
    bt.logging.info("App restarted due to update")

    python = sys.executable
    os.execl(python, python, *sys.argv)


def try_update_packages():
    bt.logging.info("Try updating packages...")

    try:
        repo = git.Repo(search_parent_directories=True)
        repo_path = repo.working_tree_dir

        requirements_path = os.path.join(repo_path, "requirements.txt")

        python_executable = sys.executable
        subprocess.check_call(
            [python_executable], "-m", "pip", "install", "-r", requirements_path
        )
        bt.logging.info("Finished updating packages.")

    except Exception as e:
        bt.logging.info(f"Updating packages failed {e}")


def try_update():
    try:
        if check_version_updated() == True:
            bt.logging.info("Updating local files...")
            if update_repo() == True:
                try_update_packages()
                restart_app()

    except Exception as e:
        bt.logging.info(f"Try updating failed {e}")
