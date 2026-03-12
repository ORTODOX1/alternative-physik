"""
Shared fixtures for LENR test suite.
"""
import sys
import os
import pytest

# Ensure python/ and project root are on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')

for p in [PYTHON_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def physics_engine():
    from physics_engine import PhysicsEngine
    return PhysicsEngine()


@pytest.fixture
def cherepanov_engine():
    from cherepanov_engine import CherepanovEngine
    return CherepanovEngine()


@pytest.fixture
def data_generator():
    from data_generator_v2 import LENRDataGeneratorV2
    return LENRDataGeneratorV2()


@pytest.fixture
def alloy_predictor():
    from alloy_predictor import AlloyPredictor
    return AlloyPredictor()


@pytest.fixture
def barrier_falsification():
    from barrier_falsification import BarrierFalsification
    return BarrierFalsification()
