from .learning_model import (
    Gawron,
    Culo,
    WeightedAverage
)

from ..keychain import Keychain as kc

def get_learning_model(params, initial_knowledge):
    """
    Returns a learning model based on the provided parameters.

    Parameters:
    params (dict): A dictionary containing model parameters.

    Returns:
    BaseLearningModel: A learning model object.
    """
    model = params[kc.MODEL]
    if model == kc.GAWRON:
        return Gawron(params, initial_knowledge)
    elif model == kc.CULO:
        return Culo(params, initial_knowledge)
    elif model == kc.W_AVG:
        return WeightedAverage(params, initial_knowledge)
    else:
        raise ValueError('[MODEL INVALID] Unrecognized model: ' + model)