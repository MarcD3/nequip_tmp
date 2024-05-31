from typing import List

import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class SumTwoModels(GraphModuleMixin, torch.nn.Module):
    """Module that sums the energy predictions of two different full models.

    Warning:
        Do not rely on which model's results will be returned in keys other
        than those in `sum_in_keys` when both `use_model1` and `use_model2`
        are true.

    Args:
        use_model1:  whether to include the model1 term
        use_model2:  whether to include the model2 term
    """

    use_model1: bool
    use_model2: bool
    sum_in_keys: List[str]

    def __init__(
        self,
        sum_in_keys: List[str],
        model1: GraphModuleMixin,
        model2: GraphModuleMixin,
        use_model1: bool = True,
        use_model2: bool = True,
        irreps_in=None,
    ) -> None:
        super().__init__()
        self._init_irreps(irreps_out=model2.irreps_out, irreps_in=irreps_in)
        assert use_model1 or use_model2, "must use at least one of model1 and model2"
        self.sum_in_keys = sum_in_keys
        for k in self.sum_in_keys:
            assert model1.irreps_out[k] == model2.irreps_out[k]
        self.model1_model = model1 if use_model1 else None
        self.model2_model = model2 if use_model2 else None
        self.use_model1, self.use_model2 = use_model1, use_model2

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.use_model1 and self.use_model2:
            # run model1
            data_model1 = self.model1_model(data.copy())
            # We only need keys we're going to sum into model2's output dict
            # let other output tensors go out of scope, in case
            data_model1 = {k: data_model1[k] for k in self.sum_in_keys}
            # run model2
            data_model2 = self.model2_model(data.copy())
            # sum the contributions of model1 into model2's output
            for k in self.sum_in_keys:
                data_model2[k] = data_model2[k] + data_model1[k]
            return data_model2
        elif self.use_model2:
            # self.use_model1 is False here, so just run model2
            return self.model2_model(data)
        elif self.use_model1:
            # self.use_model2 is False here, so just run model1
            return self.model1_model(data)
        else:
            # impossible
            assert False
