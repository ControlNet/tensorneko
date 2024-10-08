from typing import OrderedDict, Union
from pathlib import Path

import torch

from tensorneko_util.io._path_conversion import _path2str
from ...util import Device


class WeightReader:
    """WeightReader for read model weights (checkpoints, state_dict, etc)."""

    @classmethod
    def of_pt(cls, path: Union[str, Path], map_location: Device = "cpu") -> OrderedDict[str, torch.Tensor]:
        """
        Reads PyTorch model weights from a `.pt` or `.pth` file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the `.pt` or `.pth` file.
            map_location (:class:`torch.device` | ``str``): The location to load the model weights. Default: "cpu"

        Returns:
            :class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]: The model weights.
        """
        path = _path2str(path)
        return torch.load(path, map_location=map_location)

    @classmethod
    def of_ckpt(cls, path: Union[str, Path], map_location: Device = "cpu") -> OrderedDict[str, torch.Tensor]:
        """
        Reads PyTorch model weights from a `.ckpt` file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the `.ckpt` file.
            map_location (:class:`torch.device` | ``str``): The location to load the model weights. Default: "cpu"

        Returns:
            :class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]: The model weights.
        """
        path = _path2str(path)
        return torch.load(path, map_location=map_location)["state_dict"]

    @classmethod
    def of_safetensors(cls, path: Union[str, Path], map_location: str = "cpu") -> OrderedDict[str, torch.Tensor]:
        """
        Reads model weights from a `.safetensors` file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the `.safetensors` file.
            map_location (``str``): The location to load the model weights. Default: "cpu"

        Returns:
            :class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]: The model weights.
        """
        path = _path2str(path)
        import safetensors
        from collections import OrderedDict
        tensors = OrderedDict()
        with safetensors.safe_open(path, framework="pt", device=map_location) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors

    @classmethod
    def of(cls, path: Union[str, Path], map_location: Device = "cpu") -> OrderedDict[str, torch.Tensor]:
        """
        Reads model weights from a file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the file.
            map_location (:class:`torch.device` | ``str``): The location to load the model weights. Default: "cpu"

        Returns:
            :class:`collections.OrderedDict`[``str``, :class:`torch.Tensor`]: The model weights.
        """
        path = _path2str(path)
        if path.endswith(".pt") or path.endswith(".pth"):
            return cls.of_pt(path, map_location)
        elif path.endswith(".ckpt"):
            return cls.of_ckpt(path, map_location)
        elif path.endswith(".safetensors"):
            if isinstance(map_location, torch.device):
                map_location = str(map_location)
            return cls.of_safetensors(path, map_location)
        else:
            raise ValueError("Unknown file type. Supported types: .pt, .pth, .ckpt, .safetensors")

    def __new__(cls, path: Union[str, Path], map_location: Device = "cpu") -> OrderedDict[str, torch.Tensor]:
        """Alias of :meth:`~tensorneko.io.weight.weight_reader.WeightReader.of`."""
        path = _path2str(path)
        return cls.of(path, map_location)
