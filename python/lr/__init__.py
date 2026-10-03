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

from . import transforms  # noqa: F401
