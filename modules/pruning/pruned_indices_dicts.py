from typing import Union

import numpy as np


class LayerPrunedIndicesDict(dict):
    def __setitem__(self, key: str, value: Union[list, np.ndarray]) -> None:
        if key not in ["input", "output"]:
            raise ValueError("Key must be either 'input' or 'output'")
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError("Value must be a list or numpy array")
        super().__setitem__(key, value)


class BlockPrunedIndicesDict(dict):
    def __setitem__(
        self, key: str, value: Union[LayerPrunedIndicesDict, "BlockPrunedIndicesDict"]
    ) -> None:
        if not isinstance(value, (LayerPrunedIndicesDict, BlockPrunedIndicesDict)):
            raise ValueError(
                "Value must be an instance of LayerPrunedIndicesDict or BlockPrunedIndicesDict"
            )
        super().__setitem__(key, value)


class ModelPrunedIndicesBag(dict):
    def __setitem__(
        self, key: str, value: Union[LayerPrunedIndicesDict, BlockPrunedIndicesDict]
    ) -> None:
        if not isinstance(value, (LayerPrunedIndicesDict, BlockPrunedIndicesDict)):
            raise TypeError(
                "Value must be a LayerPrunedIndicesDict or BlockPrunedIndicesDict"
            )
        super().__setitem__(key, value)
