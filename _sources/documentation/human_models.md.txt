# Human Learning and Decision-Making Models

RouteRL provides a catalog of human learning and decision-making models, including three state-of-the-art discrete choice models. These models, popular within transportation community, emulate human agents as utility maximizers, where individual utilities are influenced by individual characteristics—unlike reinforcement learning algorithms that primarily focus on cost minimization.

```{eval-rst}
.. note::
    Users can create their own human models by inheriting ``BaseLearningModel``.
```

---
### Gawron Model

```{eval-rst}
.. autoclass:: routerl.human_learning.Gawron
    :members:
    :exclude-members: _private_method
    :no-index:
```

---
### Cumulative Logit Model

```{eval-rst}
.. autoclass:: routerl.human_learning.Culo
    :members:
    :exclude-members: _private_method
    :no-index:
```

---
### Weighted Average Model

```{eval-rst}
.. autoclass:: routerl.human_learning.WeightedAverage
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

