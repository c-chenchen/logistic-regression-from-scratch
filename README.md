# Logistic Regression From Scratch (with Derivation)

This repo implements **binary logistic regression** from scratch in NumPy and explains the full derivation:
- the Bernoulli likelihood
- the cross-entropy (negative log-likelihood) loss
- the gradient
- gradient descent training
- prediction (probabilities + thresholding)

## Contents
- `Logistic_regression.py` (or notebook): NumPy implementation
- `README.md`: derivation + explanation

---

## 1) Logistic Regression Model

For a feature vector $\(x \in \mathbb{R}^d\)$, logistic regression models

\
$p(y=1 \mid x;\theta) = \sigma(z), \quad z = x^\top \theta$
\

where the sigmoid function is

\
$\sigma(z) = \frac{1}{1 + e^{-z}}$
\

### Bias / Intercept
To include an intercept term, we augment features:

\[
$x_b = \begin{bmatrix} 1 \\ x \end{bmatrix}, \quad
\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_d \end{bmatrix}$
\]

so that

\[
$z = x_b^\top \theta = \theta_0 + \sum_{j=1}^{d}\theta_j x_j$
\]

In code this is done by adding a column of ones to $\(X\)$.

---

## 2) Probabilistic Assumption (Bernoulli)

For each sample \(i\), the label $\(y_i \in \{0,1\}\)$ is modeled as:

\[
$Y_i \mid x_i, \theta \sim \text{Bernoulli}(p_i),
\quad p_i = \sigma(x_i^\top \theta)$
\]

The Bernoulli PMF is:

\[
$P(y_i \mid x_i, \theta)
= p_i^{y_i}(1-p_i)^{1-y_i}$
\]

---

## 3) Likelihood, Log-Likelihood, and Loss

Assuming i.i.d. samples, the likelihood is:

\[
$\mathcal{L}(\theta)
= \prod_{i=1}^{m} p_i^{y_i}(1-p_i)^{1-y_i}$
\]

Taking logs:

\[
$\ell(\theta)
= \sum_{i=1}^{m}\left[
y_i\log(p_i) + (1-y_i)\log(1-p_i)
\right]$
\]

We minimize the **negative average log-likelihood** (binary cross-entropy):

\[
$J(\theta)
= -\frac{1}{m}\sum_{i=1}^{m}\left[
y_i\log(p_i) + (1-y_i)\log(1-p_i)
\right]$
\]

---

## 4) Gradient Derivation (Key Result)

Let $\(z_i = x_i^\top \theta\) and \(p_i = \sigma(z_i)\)$. Using:

\[
$\frac{d}{dz}\sigma(z)=\sigma(z)(1-\sigma(z))$
\]

the gradient of the loss simplifies to:

\[
$\nabla_\theta J(\theta)
= \frac{1}{m}\sum_{i=1}^{m}(p_i - y_i)\,x_i$
\]

In matrix form:

\[
$\nabla_\theta J(\theta)
= \frac{1}{m} X^\top(\sigma(X\theta)-y)$
\]

This is the expression implemented in the repo.

---

## 5) Optimization (Gradient Descent)

Gradient descent iteratively updates:

\[
$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$
\]

where $\(\alpha\)$ is the learning rate.

---

## 6) Prediction

### Probability prediction
\[
$\hat{p} = \sigma(X\theta)$
\]

### Class prediction (thresholding)
\[
$\hat{y} =
\begin{cases}
1 & \hat{p} \ge 0.5 \\
0 & \hat{p} < 0.5
\end{cases}$
\]

> Note: the threshold can be changed depending on the application (precision/recall tradeoff, class imbalance, etc.).

---

# Code Walkthrough (Function-by-Function)

### `sigmoid(z)`
Implements the sigmoid function:
\[
$\sigma(z)=\frac{1}{1+e^{-z}}$
\]

### `calculate_gradient(theta, X, y)`
Computes the gradient:
\[
$\nabla_\theta J(\theta)=\frac{1}{m}X^\top(\sigma(X\theta)-y)$
\]

### `gradient_descent(X, y, alpha, num_iter, tol)`
1. Builds $\(X_b = [\mathbf{1}\ \ X]\)$ to include bias/intercept  
2. Initializes \(\theta=0\)  
3. Repeats:
   - compute gradient
   - update \(\theta\)
   - stop early when $\(\|\nabla J(\theta)\| < \text{tol}\)$

### `predict_proba(X, theta)`
Returns probabilities:
\[
$\hat{p}=\sigma(X_b\theta)$
\]

### `predict(X, theta, threshold)`
Returns 0/1 predictions based on the chosen threshold.

---

## How to Run

Example (breast cancer dataset):

1. Standardize features (important for gradient descent stability)
2. Train with gradient descent
3. Evaluate accuracy

---

## Notes / Improvements
- Add the explicit loss computation to monitor convergence
- Add regularization (L2 / L1)
- Tune learning rate and iterations
- Compare against `sklearn.linear_model.LogisticRegression`

---

## References
- Bernoulli likelihood and maximum likelihood estimation
- Binary cross-entropy / logistic loss
- Gradient derivation for logistic regression
