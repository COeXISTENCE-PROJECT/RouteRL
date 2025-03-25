from routerl.keychain import Keychain as kc

from routerl.human_learning import (
    GawronModel,
    GeneralModel,
    WeightedModel,
    RandomModel,
    AONModel
)

def get_learning_model(params, initial_knowledge):
    """Returns a learning model based on the provided parameters.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (Any): A dictionary containing initial knowledge.
    Returns:
        BaseLearningModel: A learning model object.
    Raises:
        ValueError: If model is unknown.
    """

    model = params[kc.MODEL]
    if model == kc.GAWRON:
        return GawronModel(params, initial_knowledge)
    if model == kc.GENERAL_MODEL:
        return GeneralModel(params, initial_knowledge)
    elif model == kc.WEIGHTED:
        return WeightedModel(params, initial_knowledge)
    elif model == kc.RANDOM:
        return RandomModel(params, initial_knowledge)
    elif model == kc.AON:
        return AONModel(params, initial_knowledge)
    else:
        raise ValueError('[MODEL INVALID] Unrecognized model: ' + model)