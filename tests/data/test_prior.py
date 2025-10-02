from tspfn.data.prior import PriorConfig
from . import get_test_config_path


def test_seeding():
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior()
    prior2 = prior_config.sample_prior()
    assert prior1 != prior2
    
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    second_prior1 = prior_config.sample_prior()
    second_prior2 = prior_config.sample_prior()
    
    assert prior1 == second_prior1
    assert prior2 == second_prior2