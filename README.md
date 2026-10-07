# Deep Rho-Odds Mixture (DRM)

A survival analysis framework combining **Deep Cox Mixtures (DCM)** with the **Proportional Rho-Odds (PRO)** model.

## Overview

Survival analysis deals with time-to-event data where the event of interest may not be observed for every individual due to censoring. Traditional survival models such as the Cox Proportional Hazards model rely on the proportional hazards assumption, which may not adequately represent heterogeneous populations or complex relationships between covariates and survival.

This project investigates a **Deep Rho-Odds Mixture (DRM)** model that combines two ideas:

- **Deep Cox Mixtures (DCM):** represents the population as a mixture of latent subgroups whose assignments are learned using neural networks.
- **Proportional Rho-Odds (PRO):** provides a generalized survival model controlled by the parameter $\rho$, encompassing both proportional hazards and proportional odds formulations.

The proposed model replaces the Cox components of DCM with Proportional Rho-Odds components.

---

## Model Architecture

The proposed model consists of three main components:

1. **Representation Network**
   - A neural network learns a representation of the input covariates.
   - Denoted by $\phi_\theta(\mathbf{x})$.

2. **Gating Network**
   - Produces the probability of assigning an individual to each of the $K$ latent mixture components.
   - A softmax transformation ensures that the mixture probabilities sum to one.

3. **Rho-Odds Components**
   - Each latent component follows a Proportional Rho-Odds survival model.
   - Each component has its own regression coefficients and baseline function.

The overall survival function is

$$
S(t \mid \mathbf{x})
=
\sum_{k=1}^{K}
\pi_k(\mathbf{x})S_k(t \mid \mathbf{x}).
$$

For component $k$,

$$
S_k(t \mid \mathbf{x})
=
\left[
1+
\rho A_k(t)
\exp\left(
\beta_k^\top\phi_\theta(\mathbf{x})
\right)
\right]^{-1/\rho}.
$$

Here:

- $K$ is the number of mixture components.
- $\pi_k(\mathbf{x})$ is the neural-network-derived mixture probability.
- $\rho$ is the Proportional Rho-Odds parameter.
- $A_k(t)$ is the component-specific baseline function.
- $\beta_k$ represents the component-specific covariate effects.
- $\phi_\theta(\mathbf{x})$ is the learned representation of the covariates.

---

## Mixture Hazard

The hazard of the mixture is not simply the weighted average

$$
\sum_{k=1}^{K}\pi_k(\mathbf{x})\lambda_k(t\mid\mathbf{x}).
$$

Instead, it is

$$
\lambda(t\mid\mathbf{x})
=
\frac{
\sum_{k=1}^{K}
\pi_k(\mathbf{x})
S_k(t\mid\mathbf{x})
\lambda_k(t\mid\mathbf{x})
}{
\sum_{k=1}^{K}
\pi_k(\mathbf{x})
S_k(t\mid\mathbf{x})
}.
$$

This occurs because the probability of belonging to a component changes after conditioning on surviving until time $t$. The effective component weight becomes

$$
P(Z=k\mid T>t,\mathbf{x})
=
\frac{
\pi_k(\mathbf{x})S_k(t\mid\mathbf{x})
}{
\sum_j\pi_j(\mathbf{x})S_j(t\mid\mathbf{x})
}.
$$

Thus, components with higher survival probabilities receive greater influence on the overall hazard at later times.

---

## Proportional Rho-Odds Model

The Proportional Rho-Odds model is defined by

$$
\log
\left[
\frac{
S(t\mid\mathbf{x})^{-\rho}-1
}{
\rho
}
\right]
=
\alpha(t)+\beta^\top\mathbf{x}.
$$

Equivalently,

$$
S(t\mid\mathbf{x})
=
\left[
1+
\rho\exp\left\{
\alpha(t)+\beta^\top\mathbf{x}
\right\}
\right]^{-1/\rho}.
$$

The corresponding hazard is

$$
\lambda(t\mid\mathbf{x})
=
\frac{
a(t)\exp(\beta^\top\mathbf{x})
}{
1+\rho A(t)\exp(\beta^\top\mathbf{x})
},
$$

where

$$
A(t)=\exp\{\alpha(t)\},
\qquad
a(t)=A'(t).
$$

The parameter $\rho$ provides a continuum between commonly used survival models:

- $\rho \rightarrow 0$: Proportional Hazards
- $\rho = 1$: Proportional Odds

In this project, $\rho$ is treated as a hyperparameter and different values are evaluated experimentally.

---

## Likelihood

For right-censored observations $(\mathbf{x}_i,u_i,\delta_i)$, the likelihood is

$$
\mathcal{L}
=
\prod_{i=1}^{N}
f(u_i\mid\mathbf{x}_i)^{\delta_i}
S(u_i\mid\mathbf{x}_i)^{1-\delta_i}.
$$

The corresponding log-likelihood is

$$
\ell
=
\sum_{i=1}^{N}
\left[
\delta_i
\log
\left(
\sum_{k=1}^{K}
\pi_k(\mathbf{x}_i)
f_k(u_i\mid\mathbf{x}_i)
\right)
+
(1-\delta_i)
\log
\left(
\sum_{k=1}^{K}
\pi_k(\mathbf{x}_i)
S_k(u_i\mid\mathbf{x}_i)
\right)
\right].
$$

The model parameters are learned by maximizing this log-likelihood.

---

## Experiments

The experimental study focuses on investigating the effect of the main model hyperparameters:

### Number of mixture components ($K$)

Different values of $K$ are evaluated to determine how many latent survival subgroups are useful for the data.

### Rho-Odds parameter ($\rho$)

Different values of $\rho$ are evaluated to study how the generalized Rho-Odds formulation affects survival prediction.

### Baseline function

Different formulations of the time-dependent baseline component are investigated as part of the model experiments.

The experiments are performed using the **SUPPORT dataset**, a commonly used benchmark dataset for survival analysis.

---

## Dataset

The current experiments use the **SUPPORT (Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments)** dataset.

The dataset contains patient-level clinical information together with survival times and censoring indicators.

> **Note:** The dataset is not included in this repository. Please obtain it from an appropriate publicly available source and place the processed data in the expected data directory.

---

## Implementation

The model is implemented in **Python** using:

- [PyTorch](https://pytorch.org/) — neural networks and automatic differentiation
- [SciPy](https://scipy.org/) — numerical computation and optimization
- NumPy — numerical operations
- Pandas — data processing

-
