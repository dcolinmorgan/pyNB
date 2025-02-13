"""
Conftest for pytest.

This file adds the 'src' directory to sys.path so that the package can be imported
as if it were installed.
"""

import os
import sys

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 
