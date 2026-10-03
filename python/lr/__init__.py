"""Python frontend for the lightweight renderer's frame graph.

Build passes in Python on top of the engine's frame graph: create buffers and images, fill them
from numpy, write GLSL, and declare passes. The frame graph orders passes, inserts barriers, and
allocates attachment images (resizing them with the window).

    import lr
    viewer = lr.Viewer(title="demo")
    fg, res = viewer.frame_graph, viewer.resources
    ...
    viewer.run()
"""

from ._lr import *  # noqa: F401,F403
from ._lr import __doc__ as _native_doc  # noqa: F401
from ._lr import ShaderCompileError, ShaderInterfaceError, VulkanValidationError

# Report exceptions as `lr.X` (where users import them from) rather than the private `lr._lr.X`.
for _error in (ShaderCompileError, ShaderInterfaceError, VulkanValidationError):
    _error.__module__ = __name__
del _error

from . import gui, transforms  # noqa: F401

# A crash inside the native module (access violation, abort, failed runtime check) would otherwise
# end the process with no output at all; faulthandler prints the Python traceback of every thread,
# so you can see which call into lr crashed. Left alone if the application already configured it.
import faulthandler as _faulthandler

if not _faulthandler.is_enabled():
    _faulthandler.enable()
