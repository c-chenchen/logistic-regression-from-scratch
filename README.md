# Logistic Regression From Scratch (with Full Derivation + Code Explanation)

This repository implements **binary logistic regression** from scratch in NumPy and explains:
- the Bernoulli likelihood
- the cross-entropy (negative log-likelihood) loss
- the gradient derivation (step-by-step algebra)
- gradient descent training
- prediction (probabilities + thresholding)
- evaluation on the **Breast Cancer Wisconsin** dataset (via scikit-learn utilities)

---

## Contents
- `Logistic_regression.py` — NumPy implementation + evaluation script
- `README.md` — derivation + explanation of each function and pipeline step

---

## 1) Logistic Regression Model

Given a feature vector $x \in \mathbb{R}^d$, logistic regression models the probability of class 1 as:

$$
p(y=1 \mid x;\theta)=\sigma(z), \quad z=x^\top \theta
$$

where the sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

This maps any real number to $(0,1)$, so it can be interpreted as a probability.

---

## 2) Bias / Intercept

To include an intercept term, we augment the input with a constant 1:

$$
x_b = \begin{bmatrix}1\\x\end{bmatrix}, \quad
\theta = \begin{bmatrix}\theta_0\\\theta_1\\\vdots\\\theta_d\end{bmatrix}
$$

Then:

$$
z = x_b^\top \theta = \theta_0 + \sum_{j=1}^{d}\theta_j x_j
$$

In code, this is done by adding a column of ones to the feature matrix.

---

## 3) Probabilistic Assumption (Bernoulli)

For each sample $i$, we assume:

$$
Y_i \mid x_i,\theta \sim \text{Bernoulli}(p_i)
$$

with

$$
p_i = \sigma(x_{b,i}^\top \theta)
$$

The Bernoulli PMF is:

$$
P(Y_i=y_i \mid x_i,\theta)=p_i^{y_i}(1-p_i)^{1-y_i}
$$

---

## 4) Likelihood and Log-Likelihood

Assuming i.i.d. samples, the likelihood of the dataset is:

$$
\mathcal{L}(\theta)=\prod_{i=1}^{m} p_i^{y_i}(1-p_i)^{1-y_i}
$$

Taking logs:

$$
\ell(\theta)=\sum_{i=1}^{m}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

We usually **minimize** the negative average log-likelihood (binary cross-entropy):

$$
J(\theta)=-\frac{1}{m}\ell(\theta)
= -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

---

## 5) Full Gradient Derivation (Algebra)

Let:

$$
z_i = x_{b,i}^\top \theta,\quad p_i=\sigma(z_i)
$$

Start from the log-likelihood for one sample:

$$
\log P(y_i \mid x_i,\theta)
= y_i\log(p_i) + (1-y_i)\log(1-p_i)
$$

Differentiate with respect to $\theta$.

### Step 1: Use chain rule

$$
\frac{\partial}{\partial \theta}\log(p_i)
= \frac{1}{p_i}\frac{\partial p_i}{\partial \theta}
$$

$$
\frac{\partial}{\partial \theta}\log(1-p_i)
= \frac{-1}{1-p_i}\frac{\partial p_i}{\partial \theta}
$$

So:

$$
\frac{\partial}{\partial \theta}
\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
=
y_i\frac{1}{p_i}\frac{\partial p_i}{\partial \theta}
-(1-y_i)\frac{1}{1-p_i}\frac{\partial p_i}{\partial \theta}
$$

Factor out $\frac{\partial p_i}{\partial \theta}$:

$$
=
\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right]\frac{\partial p_i}{\partial \theta}
$$

### Step 2: Compute $\frac{\partial p_i}{\partial \theta}$

We have $p_i=\sigma(z_i)$ and $z_i=x_{b,i}^\top \theta$.

Sigmoid derivative:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

So:

$$
\frac{\partial p_i}{\partial \theta}
= \sigma'(z_i)\frac{\partial z_i}{\partial \theta}
= p_i(1-p_i)x_{b,i}
$$

### Step 3: Substitute back and simplify

$$
\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right]p_i(1-p_i)x_{b,i}
$$

Distribute:

$$
=
\left[y_i(1-p_i) - (1-y_i)p_i\right]x_{b,i}
$$

Expand:

$$
= (y_i - y_ip_i - p_i + y_ip_i)x_{b,i}
$$

Cancel terms:

$$
= (y_i - p_i)x_{b,i}
$$

So the gradient of the *log-likelihood* is:

$$
\nabla_\theta \ell(\theta) = \sum_{i=1}^{m}(y_i - p_i)x_{b,i}
$$

For the **loss** $J(\theta)=-\frac{1}{m}\ell(\theta)$:

$$
\nabla_\theta J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(p_i-y_i)x_{b,i}
$$

In matrix form:

$$
\nabla_\theta J(\theta)=\frac{1}{m}X_b^\top(\sigma(X_b\theta)-y)
$$

This is exactly what we implement.

---

## 6) Gradient Descent Update

Gradient descent repeatedly updates:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
$$

where $\alpha$ is the learning rate.

---

## 7) Code Explanation (Line-by-Line Mapping)

Below is the implementation used in this repository and how it matches the derivation.

---

### `sigmoid(z)`

```python
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


