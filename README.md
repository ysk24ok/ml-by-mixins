ml-by-mixins
===

## Overview

**ml-by-mixins** is my personal project where I implement various machine learning algorithms by combining loss functions, activation functions and optimizers. This library provides mixins of them and you can try any combination.

For example, if you want poisson regression with log-link function optimized by stochastic average gradient, you can write:

```py
from mlbymxn.base import BaseML
from mlbymxn.loss_functions import PoissonLossMixin
from mlbymxn.activation_functions import ExponentialActivationMixin
from mlbymxn.optimizers import SAGOptimizerMixin

class PoissonRegressionBySAG(
        BaseML, PoissonLossMixin, ExponentialActivationMixin,
        SAGOptimizerMixin):
    pass

poisson_reg_sag = PoissonRegressionBySAG(eta=0.001, max_iters=50)
poisson_reg_sag.fit(X, y)
```

Since the link function is the inverse of the activation function
(it sounds a little strange to use the terminology 'activation' for generalized linear model, but I think the link function and the activation function are related concepts),
here `ExponentialActivationMixin` is combined with `PoissonLossMixin`.
If you want poisson regression with identity-link function, all you have to do is to switch `ExponentialActivationMixin` to `IdentityActivationMixin`.

Provided mixins are as follows:

* loss function mixin
  - squared loss
  - poisson loss
  - log loss
  - hinge loss

* optimizer mixin
  - [scipy.minimize.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
  - gradient desent (GD)
  - stochastic gradient descent (SGD)
  - stochastic average gradient (SAG)
  - Newton's method
  - momentum SGD
  - RMSprop
  - AdaGrad
  - AdaDelta
  - Adam

* activation function mixin
  - identity
  - exponential
  - sigmoid
  - tanh
  - ReLU

## Formulation

|symbol|description|
|:--:|:--|
|<img src="https://latex.codecogs.com/gif.latex?m\in\mathcal{R}" title="m\in\mathcal{R}" />|the number of training samples|
|<img src="https://latex.codecogs.com/gif.latex?n\in\mathcal{R}" title="n\in\mathcal{R}" />|the number of features (including bias term)|
|<img src="https://latex.codecogs.com/gif.latex?\eta\in\mathcal{R}" title="\eta\in\mathcal{R}" />|learning rate|
|<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{x}_{i}\in\mathcal{R}^{n}" title="\boldsymbol{x}_{i}\in\mathcal{R}^{n}" />|feature vector of <img src="https://latex.codecogs.com/gif.latex?i" title="i" />-th trainig sample|
|<img src="https://latex.codecogs.com/gif.latex?y_{i}\in\mathcal{R}" title="y_{i}\in\mathcal{R}" />|target value (or label) of <img src="https://latex.codecogs.com/gif.latex?i" title="i" />-th trainig sample|
|<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}\in\mathcal{R}^{n}" title="\boldsymbol{\theta}\in\mathcal{R}^{n}" />|weight vector|

## Loss functions

Here is the basic form of loss function with L2 regularization over all training samples.  
<img src="https://latex.codecogs.com/gif.latex?l_{i}" title="l_{i}" /> is different from the type of loss functions.

<img src="https://latex.codecogs.com/gif.latex?L(\boldsymbol{\theta})=\cfrac{1}{m}\sum_{i=1}^{m}l_{i}(\boldsymbol{\theta})&plus;\cfrac{\lambda}{2}||\boldsymbol{\theta}||^{2}" title="L(\boldsymbol{\theta})=\cfrac{1}{m}\sum_{i=1}^{m}l_{i}(\boldsymbol{\theta})+\cfrac{\lambda}{2}||\boldsymbol{\theta}||^{2}" />

### Squared Loss (Linear Regression)

#### loss function

<img src="https://latex.codecogs.com/gif.latex?l_{i}(\boldsymbol{\theta})=\cfrac{1}{2}\left(\boldsymbol{x}_{i}\boldsymbol{\theta}-y_{i}\right)^{2}" title="l_{i}(\boldsymbol{\theta})=\cfrac{1}{2}\left(\boldsymbol{x}_{i}\boldsymbol{\theta}-y_{i}\right)^{2}" />

#### gradient

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial&space;l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(\boldsymbol{x}_{i}\boldsymbol{\theta}-y_{i}\right)\boldsymbol{x}_{i}" title="\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(\boldsymbol{x}_{i}\boldsymbol{\theta}-y_{i}\right)\boldsymbol{x}_{i}" />


### Poisson Loss (Poisson Regression)

#### loss function

<img src="https://latex.codecogs.com/gif.latex?l_{i}(\boldsymbol{\theta})=-\left(y_{i}\log(z_{i})-z_{i}-\log(y_{i}!)\right),\quad\left(z_{i}=\exp(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" title="l_{i}(\boldsymbol{\theta})=-\left(y_{i}\log(z_{i})-z_{i}-\log(y_{i}!)\right),\quad\left(z_{i}=\exp(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" />

#### gradient

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial&space;l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(z_{i}-y_{i}\right)\boldsymbol{x}_{i},\quad\left(z_{i}=\exp(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" title="\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(z_{i}-y_{i}\right)\boldsymbol{x}_{i},\quad\left(z_{i}=\exp(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" />

### Log Loss (Logistic Regression)

#### loss function

<img src="https://latex.codecogs.com/gif.latex?l_{i}(\boldsymbol{\theta})=-\left(y_{i}\log(z_{i})&plus;\left(1-y_{i}\right)\log\left(1-z_{i}\right)\right),\quad\left(z_{i}=sigmoid\left(\boldsymbol{x}^{(i)}\boldsymbol{\theta}\right)\right)" title="l_{i}(\boldsymbol{\theta})=-\left(y_{i}\log(z_{i})+\left(1-y_{i}\right)\log\left(1-z_{i}\right)\right),\quad\left(z_{i}=sigmoid\left(\boldsymbol{x}^{(i)}\boldsymbol{\theta}\right)\right)" />

#### gradient

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial&space;l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(z_{i}-y_{i}\right)\boldsymbol{x}_{i},\quad\left(z_{i}=sigmoid(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" title="\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial l_{i}(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}=\left(z_{i}-y_{i}\right)\boldsymbol{x}_{i},\quad\left(z_{i}=sigmoid(\boldsymbol{x}_{i}\boldsymbol{\theta})\right)" />

### Hinge Loss (Perceptron)

<img src="https://latex.codecogs.com/gif.latex?t" title="t" /> is the threshold. <img src="https://latex.codecogs.com/gif.latex?t=1" title="t=1" /> if SVM, <img src="https://latex.codecogs.com/gif.latex?t=0" title="t=0" /> if Perceptron.

#### loss function

<img src="https://latex.codecogs.com/gif.latex?l_{i}(\boldsymbol{\theta})=\max\left(0,&space;t-y_{i}\boldsymbol{x}_{i}\boldsymbol{\theta}\right)" title="l_{i}(\boldsymbol{\theta})=\max\left(0, t-y_{i}\boldsymbol{x}_{i}\boldsymbol{\theta}\right)" />

#### gradient

If <img src="https://latex.codecogs.com/gif.latex?y_{i}\boldsymbol{x}_{i}\boldsymbol{\theta}<t" title="y_{i}\boldsymbol{x}_{i}\boldsymbol{\theta}<t" />,

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial&space;l_{i}(\boldsymbol{\theta})}{\partial&space;\boldsymbol{\theta}}=-y_{i}\boldsymbol{x}_{i}" title="\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial l_{i}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=-y_{i}\boldsymbol{x}_{i}" />

otherwise

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial&space;l_{i}(\boldsymbol{\theta})}{\partial&space;\boldsymbol{\theta}}=0" title="\boldsymbol{g}_{i}(\boldsymbol{\theta})=\cfrac{\partial l_{i}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=0" />

## Optimizers

<img src="https://latex.codecogs.com/gif.latex?k" title="k" /> is the current iteration and the current weight vector <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k}" title="\boldsymbol{\theta}^{k}" /> 
will be updated to <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k&plus;1}" title="\boldsymbol{\theta}^{k+1}" />.

### Gradient Descent (GD)

GD calculates all gradients of training samples and updates theta in a batch.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k&plus;1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})&plus;\lambda\boldsymbol{\theta}^{k}\right)" title="\boldsymbol{\theta}^{k+1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})+\lambda\boldsymbol{\theta}^{k}\right)" />

### Stochastic Gradient Descent (SGD)

SGD calculates gradient of a randomly selected sample and updates theta in an online manner.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k&plus;1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})&plus;\lambda\boldsymbol{\theta}^{k}\right)" title="\boldsymbol{\theta}^{k+1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})+\lambda\boldsymbol{\theta}^{k}\right)" />

### Stochastic Average Gradient (SAG)

SAG calculates gradient of a randomly selected sample (like SGD) and updates theta in a batch (like GD).

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k&plus;1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})&plus;\lambda\boldsymbol{\theta}^{k}\right)" title="\boldsymbol{\theta}^{k+1}\leftarrow\boldsymbol{\theta}^{k}-\eta\left(\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})+\lambda\boldsymbol{\theta}^{k}\right)" />

if <img src="https://latex.codecogs.com/gif.latex?i=j" title="i=j" /> (<img src="https://latex.codecogs.com/gif.latex?j" title="j" /> is a randomly selected sample),

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})=\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})" title="\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})=\boldsymbol{g}_{i}(\boldsymbol{\theta}^{k})" />

otherwise,

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})=\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k-1})" title="\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k})=\boldsymbol{f}_{i}(\boldsymbol{\theta}^{k-1})" />

### Newton's Method

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\theta}^{k&plus;1}\leftarrow\boldsymbol{\theta}^{k}-\eta\boldsymbol{H}^{-1}\boldsymbol{g}(\boldsymbol{\theta}^{k})\quad\left(\boldsymbol{g}(\boldsymbol{\theta})=\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{g}_{i}(\boldsymbol{\theta})\right)" title="\boldsymbol{\theta}^{k+1}\leftarrow\boldsymbol{\theta}^{k}-\eta\boldsymbol{H}^{-1}\boldsymbol{g}(\boldsymbol{\theta}^{k})\quad\left(\boldsymbol{g}(\boldsymbol{\theta})=\cfrac{1}{m}\sum_{i=1}^{m}\boldsymbol{g}_{i}(\boldsymbol{\theta})\right)" />

### momentum SGD

TODO


### RMSprop

TODO


### AdaGrad

TODO


### AdaDelta

TODO


### Adam

TODO
