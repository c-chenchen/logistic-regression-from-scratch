# Logistic Regression From Scratch

This repository implements **logistic regression from scratch** using NumPy and explains the full mathematical derivation behind the algorithm.

The goal is to understand:
- where the logistic model comes from
- how the loss function is derived
- how gradient descent updates the parameters
- how predictions are made

---

## 1. Logistic Regression Model

Given an input vector \( x \in \mathbb{R}^d \), logistic regression models the probability of the positive class as:

$$
P(y = 1 \mid x) = \sigma(z)
$$

where

$$
z = x^\top \theta
$$

and the **sigmoid function** is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This maps any real number to the interval \( (0, 1) \), allowing it to be interpreted as a probability.

---

## 2. Adding the Bias Term

To allow the model to learn an intercept, we augment the feature vector:

$$
x' = [1, x_1, x_2, \dots, x_d]
$$

and the parameter vector becomes:

$$
\theta = [\theta_0, \theta_1, \dots, \theta_d]
$$

The model is now:

$$
z = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d
$$

The bias term allows the model to shift the decision boundary instead of forcing it through the origin.

---

## 3. Probabilistic Interpretation

Logistic regression assumes the target variable follows a Bernoulli distribution:

$$
Y \mid X \sim \text{Bernoulli}(p)
$$

where:

$$
p = \sigma(x^\top \theta)
$$

So the probability mass function is:

$$
P(Y = y \mid X) = p^y (1 - p)^{1 - y}
$$

---

## 4. Likelihood and Loss Function

Given a dataset of \( m \) independent samples, the likelihood is:

$$
\mathcal{L}(\theta)
= \prod_{i=1}^{m} p_i^{y_i}(1 - p_i)^{1 - y_i}
$$

Taking the negative log-likelihood gives the **binary cross-entropy loss**:

$$
\mathcal{L}(\theta)
= -\frac{1}{m}
\sum_{i=1}^{m}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
$$

This is the function minimized during training.

---

## 5. Gradient of the Loss

To optimize the parameters, we compute the gradient of the loss:

$$
\nabla_\theta \mathcal{L}
= \frac{1}{m} X^\top (\sigma(X\theta) - y)
$$

This expression tells us how to adjust the parameters to reduce the loss.

---

## 6. Gradient Descent Update Rule

The parameters are updated iteratively:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
$$

where:
- \( \alpha \) is the learning rate
- smaller values lead to slower but more stable convergence

---

## 7. Prediction

Once the model is trained, predictions are made as:

$$
\hat{p} = \sigma(X\theta)
$$

and converted to class labels using a threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 8. Interpretation

- The model outputs **probabilities**, not hard labels.
- The bias term represents the baseline log-odds.
- The sigmoid converts linear scores into valid probabilities.
- Gradient descent finds parameters that minimize cross-entropy loss.

---

## 9. Files in This Repository

- `logistic_regression.py` — full NumPy implementation  
- `README.md` — mathematical explanation and theory  

---

## 10. Summary

- Logistic regression models probabilities using a sigmoid
- Training is done via maximum likelihood estimation
- The loss function is cross-entropy
- Gradient descent is used for optimization
- The model is interpretable and probabilistic

---


