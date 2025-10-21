from typing import Dict

import torch
import torch.nn.functional as F

from roboverse_learn.il.utils.common.module_attr_mixin import ModuleAttrMixin
from roboverse_learn.il.utils.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/dp_runner.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
