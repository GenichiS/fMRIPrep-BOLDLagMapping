"""
bold_lag_mapper Package Initializer
-----------------------------------

This file serves two primary purposes:
1. It signals to Python that the `bold_lag_mapper` directory should be treated as a package,
   allowing for modular imports between the different `.py` files (e.g., `from . import core`).
2. It acts as a convenient entry point for users of the package by exposing the main
   `BOLDLagMapper` class at the top level. This allows for a cleaner import syntax.

Instead of a user needing to know the internal structure and writing:
`from bold_lag_mapper.core import BOLDLagMapper`

They can simply write the more intuitive:
`from bold_lag_mapper import BOLDLagMapper`
"""

# Define the package version as the single source of truth.
# This is a standard practice (see PEP 396) that makes it easy to check the installed version
# programmatically (`bold_lag_mapper.__version__`) and is used by packaging tools like `setuptools`.
__version__ = "2.0.0"

# Expose the main class at the package level for user convenience and a cleaner API.
# This makes the BOLDLagMapper class directly available upon importing the package.
from .core import BOLDLagMapper

