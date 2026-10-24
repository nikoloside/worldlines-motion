import unittest

from worldlines_motion.smplx_model import SMPLX, find_model


def model_or_skip() -> SMPLX:
    try:
        find_model("male")
    except FileNotFoundError as exc:
        raise unittest.SkipTest(f"no SMPL-X model: {exc}")
    return SMPLX(gender="male")
