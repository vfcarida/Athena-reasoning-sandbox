"""Pytest configuration for evaluation suite.

Imports test harness configuration from tests/conftest.py to ensure Lane 1/Lane 2
test isolation and mock imports for heavy ML libraries when running offline.
"""

from tests.conftest import *
