"""Root package info."""

import os

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)
