#!/usr/bin/env python3
"""Doxygen INPUT_FILTER: strips #include <...> (system/third-party) lines so
they don't clutter include graphs, keeping #include "..." (project) lines.
Must preserve line count exactly (Doxygen requirement for correct anchors).
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

INCLUDE_ANGLE = re.compile(r'^\s*#\s*include\s*<[^>]*>\s*$')

with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        if INCLUDE_ANGLE.match(line):
            sys.stdout.write("\n")
        else:
            sys.stdout.write(line)
