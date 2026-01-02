# Logistic Regression From Scratch

This repository implements **logistic regression from scratch** using NumPy and explains the full mathematical intuition behind it.

The goal is to understand:
- how logistic regression works
- why the sigmoid function is used
- how the loss function is derived
- how gradient descent trains the model

---

## 1. Logistic Regression Model

Logistic regression models the probability of a binary outcome.

For an input vector `x`, the model computes:

    z = xᵀθ

and applies the sigmoid function:

    σ(z) = 1 / (1 + exp(-z))

The output is interpreted as:

    P(y = 1 | x) = σ(z)

---

## 2. Bias (Intercept) Term

To allow the model to learn a baseline probability, we add a bias term.

Instead of:

    z = w1*x1 + w2*x2 + ... + wd*xd

we use:

    z = θ0 + θ1*x1 + θ2*x2 + ... + θd*xd

This is implemented by adding a column of ones to the feature matrix.

---

## 3. Probabilistic Interpretation

Each label is assumed to follow a Bernoulli distribution:

    Y ~ Bernoulli(p)

where:

    p = σ(xᵀθ)

This means:

    P(Y = 1 | X) = p
    P(Y = 0 | X) = 1 - p

---

## 4. Likelihood and Loss Function

Given a dataset of m samples, the likelihood is:

    L(θ) = ∏ p_i^y_i (1 - p_i)^(1 - y_i)

To make optimization easier, we minimize the **negative log-likelihood**:

    J(θ) = -(1/m) * Σ [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

This is also known as **binary cross-entropy loss**.

---

## 5. Gradient of the Loss

The gradient of the loss with respect to the parameters is:

    ∇J(θ) = (1/m) * Xᵀ (σ(Xθ) - y)

This tells us how to adjust the parameters to reduce the loss.

---

## 6. Gradient Descent Update

The parameters are updated iteratively using:

    θ = θ - α * ∇J(θ)

where `α` is the learning rate.

---

## 7. Prediction

After training, predictions are made as:

    p = σ(Xθ)

and class labels are assigned using a threshold:

    ŷ = 1  if p ≥ 0.5
    ŷ = 0  otherwise

---

## 8. Code Structure

- `sigmoid(z)` – computes the sigmoid function  
- `calculate_gradient()` – computes the gradient of the loss  
- `gradient_descent()` – performs optimization  
- `predict_proba()` – outputs predicted probabilities  
- `predict()` – converts probabilities to class labels  

---

## 9. Summary

- Logistic regression models probabilities using the sigmoid function  
- The bias term captures the baseline class probability  
- Training minimizes cross-entropy loss  
- Gradient descent is used to optimize parameters  
- The final output is a probability, not just a class label  

---

## 10. Files

- `logistic_regression.py` — implementation from scratch  
- `README.md` — explanation and theory  

---

If you want, I can also:
- add diagrams (decision boundary, sigmoid curve)
- convert this into a Jupyter notebook
- add comments directly inside your Python code
- or format it as a project report

Just tell me 👍



