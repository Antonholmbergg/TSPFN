import pytest

from tspfn.data.dataset import SyntheticDataset
from tspfn.data.prior import PriorConfig

from . import get_test_config_path


@pytest.mark.parametrize("seed", [37842, 42837])
def test_runs(seed):
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    torch_dataset = SyntheticDataset(prior_config, seed)
    dataset = next(torch_dataset)
    dataset2 = next(torch_dataset)
    assert dataset.shape != dataset2.shape
