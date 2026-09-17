"""HOIGPT: training, evaluation, and a lightweight public tokenizer API.

Training modules are imported explicitly so the tokenizer API only needs PyTorch.
"""

from .pointnet import PointNetEncoder
from .quantizer import EMACodebook, HOIQuantizer
from .tokenizer import HOITokenizer, HOITokens, VQVae

__all__ = [
    "EMACodebook",
    "HOIQuantizer",
    "HOITokenizer",
    "HOITokens",
    "PointNetEncoder",
    "VQVae",
]

__version__ = "0.1.0"
