# This module contains workflows which integrate the components of the other modules.

# This is the starting point for users.

import random
import string
from pathlib import Path

# ====================
# Set path for serialization
# Most workflow functions will require this
# General idea - use consistent folder tree for each interpreter instance to avoid overwriting
# ====================


global UNIQUE_
UNIQUE_ = "".join(
    random.SystemRandom().choice(string.digits + string.ascii_letters) for _ in range(4)
)


def set_global_write(
    _tempdir=rf"/mnt/c/Users/rineharn/workspace/somn_scratch_{UNIQUE_}/",
):
    """
    Setting global variables for consistent, graceful read/write

    This should be called upon initialization
    """
    global WRITE_, SCRATCH_, OUTPUT_, DESC_, STRUC_, PART_
    WRITE_, SCRATCH_, OUTPUT_, DESC_, STRUC_, PART_ = (
        _tempdir,
        f"{_tempdir}scratch/",
        f"{_tempdir}outputs/",
        f"{_tempdir}descriptors/",
        f"{_tempdir}structures/",
        f"{_tempdir}partitions/",
    )
    Path(WRITE_).mkdir(parents=True, exist_ok=True)
    Path(f"{WRITE_}partitions/").mkdir(parents=True, exist_ok=True)
    Path(f"{WRITE_}structures/").mkdir(parents=True, exist_ok=True)
    Path(f"{WRITE_}descriptors/").mkdir(parents=True, exist_ok=True)
    Path(f"{WRITE_}outputs/").mkdir(parents=True, exist_ok=True)
    Path(f"{WRITE_}scratch/").mkdir(parents=True, exist_ok=True)


set_global_write()
