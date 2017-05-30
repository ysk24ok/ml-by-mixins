ml-by-mixins
===

**ml-by-mixins** provides several machine learning algorithms
which consists of mixins of loss functions and optimizers.

* loss function mixin
  - squared loss
  - log loss

* optimizer mixin
  - gradient desent (GD)
  - stochastic gradient descent (SGD)
  - Newton's method

|symbol|description|
|:--:|:--|
|$m \in \mathcal{R}$|the number of training samples|
|$n \in \mathcal{R}$|the number of features (including bias term)|
|$\eta \in \mathcal{R}$|learning rate|
|$\boldsymbol{X} \in \mathcal{R}^{m \times n}$|matrix of training samples|
|$\boldsymbol{y} \in \mathcal{R}^{n \times 1}$|matrix of target values|