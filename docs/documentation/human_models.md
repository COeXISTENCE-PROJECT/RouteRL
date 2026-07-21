# Human Learning and Decision-Making Models

RouteRL provides a catalog of human learning and decision-making models, including three discrete choice models. These models, popular within transportation community, emulate human agents as utility maximizers, where individual utilities are influenced by individual characteristics—unlike reinforcement learning algorithms that primarily focus on cost minimization.

```{eval-rst}
.. note::
    Users can create their own human models by inheriting ``BaseLearningModel``.
```

---
### Random Model

```{eval-rst}
.. autoclass:: routerl.human_learning.RandomModel
    :members:
    :exclude-members: _private_method
    :no-index:
```

---
### Gawron Model

```{eval-rst}
.. autoclass:: routerl.human_learning.GawronModel
    :members:
    :exclude-members: _private_method
    :no-index:
```

---
### Weighted Average Model

```{eval-rst}
.. autoclass:: routerl.human_learning.WeightedModel
    :members:
    :exclude-members: _private_method
    :no-index:
```

---
### Base Learning Model

```{eval-rst}
.. autoclass:: routerl.human_learning.learning_model.BaseLearningModel
    :members:
    :exclude-members: _private_method
    :no-index:
```

