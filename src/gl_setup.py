"""
Patch ctypes.CDLL so that moderngl can find libEGL.so / libGL.so on Linux
systems where only versioned .so files are installed (no -dev packages).

moderngl's _moderngl.DefaultLoader does:
    ctypes.CDLL("libEGL.so")   # fails without the -dev symlink
    ctypes.CDLL("libGL.so")    # same

This module monkey-patches ctypes.CDLL (before moderngl is imported) to
transparently redirect un-versioned names to their versioned counterparts.
"""

import ctypes
import os
import sys

_FALLBACKS = {
    "libEGL.so": "libEGL.so.1",
    "libGL.so": "libGL.so.1",
}

_LIB_SEARCH_DIRS = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
]

_original_CDLL_init = ctypes.CDLL.__init__


def _patched_CDLL_init(self, name, *args, **kwargs):
    try:
        _original_CDLL_init(self, name, *args, **kwargs)
    except OSError:
        if not isinstance(name, str) or name not in _FALLBACKS:
            raise
        # Try versioned fallback from system lib dirs.
        versioned = _FALLBACKS[name]
        for d in _LIB_SEARCH_DIRS:
            path = os.path.join(d, versioned)
            if os.path.exists(path):
                _original_CDLL_init(self, path, *args, **kwargs)
                return
        raise


if sys.platform.startswith("linux"):
    ctypes.CDLL.__init__ = _patched_CDLL_init
