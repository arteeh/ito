import sys

import moderngl


def current_context() -> moderngl.Context:
    """Wrap the window's current OpenGL 4.3 context.

    On Linux, libgl=None lets glcontext find the runtime libGL without development symlinks;
    on Windows and macOS glcontext must use its own platform default.
    """
    if sys.platform == "linux":
        return moderngl.create_context(require=430, libgl=None)
    return moderngl.create_context(require=430)
