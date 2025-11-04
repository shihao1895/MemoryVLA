from abc import ABC, abstractmethod
from typing import List, Dict
import torch


class Tokenization(ABC):
    """ Tokenize the language for LLM
    """
    @abstractmethod
    def __call__(self, conversations: List[Dict],
                 has_image: bool) -> dict[str, torch.Tensor]:
        pass


class DummyTokenization(Tokenization):
    def __call__(self, conversations: List[Dict],
                 has_image: bool) -> dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor([0]),
            'labels': torch.tensor([0])
        }
