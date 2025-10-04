import pytest

from tspfn.data.prior import PriorConfig

from . import get_test_config_path


@pytest.mark.parametrize("seed", [37842, 42837, 6374])
def test_seeding(seed):
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior(seed)
    prior1_1 = prior_config.sample_prior(seed)
    prior2 = prior_config.sample_prior(seed * 2)
    assert prior1 != prior2
    assert prior1 == prior1_1

    second_prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    second_prior1 = second_prior_config.sample_prior(seed)
    second_prior2 = second_prior_config.sample_prior(seed * 2)

    assert prior1 == second_prior1
    assert prior2 == second_prior2
