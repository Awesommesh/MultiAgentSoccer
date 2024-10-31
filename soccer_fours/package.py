import logging
import os
from pathlib import Path
import platform
import shutil
import stat
import zipfile
import requests, zipfile, io



TRAINING_ENV_PATH = "UnityEnvBuilds/Soccer4.app"
ROLLOUT_ENV_PATH = "UnityEnvBuilds/Soccer4.app"


def check_package():
    """
    Checks if the package is installed and, if not, installs it according to the
    current platform.
    """

    if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():
        logging.info(
            f"BINARY ENVS NOT FOUND!"
        )
        print("HERE HERE")
    else:
        logging.debug(
            f"Binary envs found in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'"
        )
    logging.debug("Env verification done.")
