from typing import Union, Dict

import numpy as np


class SubmodelLayerParamIndicesDict(dict):
    def __init__(self, initial_dict: Dict[str, np.ndarray] = None):
        if initial_dict is not None:
            for key, value in initial_dict.items():
                self.__setitem__(key, value)

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if key not in ["in", "out"]:
            raise ValueError("Key must be either 'in' or 'out'")
        if not isinstance(value, np.ndarray):
            raise ValueError("Value must be an instance of numpy.ndarray")
        super().__setitem__(key, value)


class SubmodelBlockParamIndicesDict(dict):
    def __init__(self, initial_dict: Dict[str, Union[SubmodelLayerParamIndicesDict, "SubmodelBlockParamIndicesDict"]] = None):
        if initial_dict is not None:
            for key, value in initial_dict.items():
                self.__setitem__(key, value)

    def __setitem__(
        self,
        key: str,
        value: Union[SubmodelLayerParamIndicesDict, "SubmodelBlockParamIndicesDict"],
    ) -> None:
        if not isinstance(
            value, (SubmodelLayerParamIndicesDict, SubmodelBlockParamIndicesDict)
        ):
            raise ValueError(
                "Value must be an instance of SubmodelLayerParamIndicesDict or SubmodelBlockParamIndicesDict"
            )
        super().__setitem__(key, value)
