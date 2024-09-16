"""Tests for the metric module"""

import os
import sys
import torch
ROOT_FOLDER = "bertaphore"
p = os.getcwd()
while os.path.basename(p) != ROOT_FOLDER:
    p = os.path.dirname(p)
sys.path.insert(0, p)
from modules.metric import AttentionalBurden  # pylint: disable=wrong-import-position

class TestBasic:
    """Basic test class"""
    def setup_method(self):
        """Setup method, called before each test"""

    def test_basic(self):
        """Basic test"""
        assert 1 == 1

class TestAttentionalBurden:
    """Test class for AttentionalBurden"""
    def setup_method(self):
        """Setup method, called before each test"""

    def test_computeattentionCost(self):
        """Test the compute_attentionCost method"""
        t1 = torch.tensor([[[[1., 1., 3.],
                    [0., 1., 1.],
                    [0., 0., 1.]]]])
        t2 = t1
        t_top = torch.cat((t1, torch.zeros_like(t1)),dim=3)
        t_bot = torch.cat((torch.zeros_like(t1), t2),dim=3)
        t = torch.cat((t_top, t_bot),dim=2)
        b = AttentionalBurden.compute_attentionCost(t)
        b1 = AttentionalBurden.compute_attentionCost(t1)
        assert b==b1
