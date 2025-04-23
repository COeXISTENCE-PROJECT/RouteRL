from routerl.keychain import Keychain as kc

from routerl.human_learning import (
    AON,
    Gawron,
    Culo,
    Random,
    WeightedAverage
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
    if model == kc.AON:
        return AON(params, initial_knowledge)
    elif model == kc.GAWRON:
        return Gawron(params, initial_knowledge)
    elif model == kc.CULO:
        return Culo(params, initial_knowledge)
    elif model == kc.RANDOM:
        return Random(params, initial_knowledge)
    elif model == kc.W_AVG:
        return WeightedAverage(params, initial_knowledge)
    else:
        raise ValueError('[MODEL INVALID] Unrecognized model: ' + model)