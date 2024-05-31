from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, SumTwoModels
from nequip.model import model_from_config
from nequip.utils import Config


def DeploySumOfTwoModels(
    config: Config,
    initialize: bool,
    deploy: bool,
    dataset=None,
) -> GraphModuleMixin:
    """Deploy the sum of two complete models.

    Warning:
        Deployment metadata, such as model cutoff, may confict between model1 and model2!
        DeploySumOfTwoModels DOES NOT CHECK their consistency, and incorrect or unexpected
        behavior may result.
    """
    assert (
        deploy
    ), "DeploySumOfTwoModels is only meant to deploy together the sum of two models trained separately."
    assert not initialize

    sum_in_keys = config.get("sum_in_keys")

    configs = {k: Config.from_file(config[k + "_config"]) for k in ("model1", "model2")}
    for k, c in configs.items():
        c.update(config.get(k + "_overrides", {}))

    models = {
        prefix: model_from_config(
            config=c, initialize=initialize, dataset=dataset, deploy=deploy
        ).model
        for prefix, c in configs.items()
    }

    # Update the saved config for deployment with options from the individual models
    # to populate things like r_max, dtype settings, etc.
    # WARNING !!:  this ASSUMES compatibility of the models on all important options,
    # and fills the rest of the saved config with an arbitrary combination of the two!
    for k, c in configs.items():
        config.update(c)

    return SumTwoModels(
        sum_in_keys=sum_in_keys,
        model1=models["model1"],
        model2=models["model2"],
    )
