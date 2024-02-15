from typing import Dict

import torch


class WeightWriter:
    """WeightWriter for write model weights (checkpoints, state_dict, etc)."""

    @classmethod
    def to_pt(cls, path: str, weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes PyTorch model weights to a `.pt` or `.pth` file.

        Args:
            path (``str``): The path of the `.pt` or `.pth` file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        torch.save(weights, path)

    @classmethod
    def to_safetensors(cls, path: str, weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes model weights to a `.safetensors` file.

        Args:
            path (``str``): The path of the `.safetensors` file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        import safetensors.torch
        safetensors.torch.save_file(weights, path)

    @classmethod
    def to(cls, path: str, weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes model weights to a file.

        Args:
            path (``str``): The path of the file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        file_type = path.split(".")[-1]

        if file_type == "pt":
            cls.to_pt(path, weights)
        elif file_type == "safetensors":
            cls.to_safetensors(path, weights)
        else:
            raise ValueError("Unknown file type. Supported types: .pt, .safetensors")
