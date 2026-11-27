"""pytest configuration for the slabitz test suite (T02).

Registers the markers used by tests/test_suite.py so `sbz pytest -m "not slow"`
(and any other marker expression) is accepted without a warning:

  gpu   the script needs a CUDA device (a context is created via sbz_dctx_create_cuda)
  slow  the script takes more than 60 s to run (measured once, in test_suite.py)
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: the script needs a CUDA device")
    config.addinivalue_line("markers", "slow: the script takes more than 60 s")
