# Logistic Regression From Scratch

This repository implements **logistic regression from scratch** and explains the full mathematical derivation behind the model.

---

## 1) Logistic Regression Model

For a feature vector \( x \in \mathbb{R}^d \), logistic regression models the probability of the positive class as:

$$
P(y = 1 \mid x) = \sigma(z)
$$

where

$$
z = x^\top \theta
$$

and the sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

## 2) Bias Term

To include an intercept term, we augment the feature vector:

$$
x' = [1, x_1, x_2, \dots, x_d]
$$

and the parameter vector becomes:

$$
\theta = [\theta_0, \theta_1, \dots, \theta_d]
$$

This allows the model to learn a baseline probability even when all features are zero.

---

## 3) Probabilistic Interpretation

The target variable is modeled as a Bernoulli random variable:

$$
Y \mid X \sim \text{Bernoulli}(p)
$$

where

$$
p = \sigma(x^\top \theta)
$$

Thus,

$$
P(Y = 1 \mid X) = \sigma(x^\top \theta)
$$

and

$$
P(Y = 0 \mid X) = 1 - \sigma(x^\top \theta)
$$

---

## 4) Likelihood and Loss Function

The likelihood of the dataset is:

$$
\mathcal{L}(\theta)
= \prod_{i=1}^{m} p_i^{y_i}(1 - p_i)^{1 - y_i}
$$

Taking the negative log-likelihood gives the loss:

$$
\mathcal{L}(\theta)
= -\frac{1}{m} \sum_{i=1}^{m}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
$$

This is known as the **binary cross-entropy loss**.

---

## 5) Gradient of the Loss

The gradient with respect to the parameters is:

$$
\nabla_\theta \mathcal{L}
= \frac{1}{m} X^\top (\sigma(X\theta) - y)
$$

---

## 6) Gradient Descent Update

The parameters are updated using:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
$$

where \( \alpha \) is the learning rate.

---

## 7) Prediction

Predicted probabilities:

$$
\hat{p} = \sigma(X\theta)
$$

Predicted class labels:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 8) Summary

- Logistic regression models the probability of a binary outcome  
- The sigmoid maps real values to probabilities  
- The bias term captures the baseline likelihood  
- Training minimizes cross-entropy via gradient descent  

---

## 9) Files

- `logistic_regression.py` — implementation from scratch  
- `README.md` — mathematical explanation and derivation
