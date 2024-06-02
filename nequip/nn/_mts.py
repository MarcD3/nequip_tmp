import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class MTSEnergySum(GraphModuleMixin, torch.nn.Module):
    """Module that sums the energy predictions of two different models."""

    use_inner: bool
    use_outer: bool

    def __init__(
        self,
        inner_model: GraphModuleMixin,
        outer_model: GraphModuleMixin,
        use_inner: bool = True,
        use_outer: bool = True,
        irreps_in=None,
    ) -> None:
        super().__init__()
        self._init_irreps(irreps_out=outer_model.irreps_out, irreps_in=irreps_in)
        assert use_inner or use_outer, "must use at least one of inner and outer"
        assert AtomicDataDict.FORCE_KEY not in inner_model.irreps_out
        assert AtomicDataDict.FORCE_KEY not in outer_model.irreps_out
        self.inner_model = inner_model if use_inner else None
        self.outer_model = outer_model if use_outer else None
        self.use_inner, self.use_outer = use_inner, use_outer

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.use_inner and self.use_outer:
            eng_inner = self.inner_model(data.copy())[AtomicDataDict.TOTAL_ENERGY_KEY]
            data_outer = self.outer_model(data.copy())
            eng_outer = data_outer[AtomicDataDict.TOTAL_ENERGY_KEY]
            data_outer[AtomicDataDict.TOTAL_ENERGY_KEY] = eng_inner + eng_outer
            return data_outer
        elif self.use_outer:
            return self.outer_model(data)
        elif self.use_inner:
            return self.inner_model(data)
        else:
            assert False
