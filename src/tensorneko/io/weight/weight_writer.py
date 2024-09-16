from typing import Dict, Union
from pathlib import Path

import torch

from tensorneko_util.io._path_conversion import _path2str
from ...util import Device

class WeightWriter:
    """WeightWriter for write model weights (checkpoints, state_dict, etc)."""

    @classmethod
    def to_pt(cls, path: Union[str, Path], weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes PyTorch model weights to a `.pt` or `.pth` file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the `.pt` or `.pth` file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        path = _path2str(path)
        torch.save(weights, path)

    @classmethod
    def to_safetensors(cls, path: Union[str, Path], weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes model weights to a `.safetensors` file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the `.safetensors` file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        path = _path2str(path)
        import safetensors.torch
        safetensors.torch.save_file(weights, path)

    @classmethod
    def to(cls, path: Union[str, Path], weights: Dict[str, torch.Tensor]) -> None:
        """
        Writes model weights to a file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the file.
            weights (:class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]): The model weights.
        """
        path = _path2str(path)
        file_type = path.split(".")[-1]

        if file_type in ("pt", "pth"):
            cls.to_pt(path, weights)
        elif file_type == "safetensors":
            cls.to_safetensors(path, weights)
        else:
            raise ValueError("Unknown file type. Supported types: .pt, .safetensors")

    def __new__(cls, path: Union[str, Path], weights: Dict[str, torch.Tensor]) -> None:
        """Alias of :meth:`~tensorneko.io.weight.weight_writer.WeightWriter.to`."""
        path = _path2str(path)
        return cls.to(path, weights)
