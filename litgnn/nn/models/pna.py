from torch import Tensor
from torch_geometric.nn import PNA as _PyG_PNA
from torch_geometric.nn import PNAConv


class PNA(_PyG_PNA):

    deg: Tensor = None

    @classmethod
    def compute_degree(cls, dataloader) -> None:

        cls.deg = PNAConv.get_degree_histogram(dataloader)

    def __init__(self, **kwargs) -> None:

        assert PNA.deg is not None, \
            "`deg` cannot be None. Please call `PNA.compute_degree` before instantiating the class."
        kwargs["deg"] = PNA.deg
        super().__init__(**kwargs)
