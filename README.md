# Deep Rho-Odds Mixture (DRM)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep survival analysis framework combining **Deep Cox Mixtures (DCM)** with the **Proportional Rho-Odds (PRO)** model to accommodate heterogeneous populations and non-proportional hazard dynamics.

---

## Overview

Survival analysis models time-to-event data where observations are subject to right-censoring. Traditional models, such as the Cox Proportional Hazards model, rely on the strict proportional hazards assumption, which often fails in heterogeneous populations with complex covariate interactions.

The **Deep Rho-Odds Mixture (DRM)** addresses this by integrating two methodologies:

* **Deep Cox Mixtures (DCM):** Models subpopulation heterogeneity by learning latent group assignments via deep neural networks.
* **Proportional Rho-Odds (PRO):** A unified survival formulation parameterized by $\rho$, offering a flexible continuum between proportional hazards ($\rho \to 0$) and proportional odds ($\rho = 1$).

By replacing DCM’s standard Cox components with Proportional Rho-Odds components, DRM gains flexibility in capturing non-proportional survival patterns across latent subgroups.

---

## Model Architecture

The DRM pipeline processes input covariates through three unified modules:

```
                  ┌──────────────────────────────┐
                  │ Input Covariates (x)         │
                  └──────────────┬───────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      ┌─────────────────────────┐ ┌─────────────────────────┐
      │ Representation Network  │ │     Gating Network      │
      │    ϕ_θ(x) Embedding     │ │      Softmax π_k(x)     │
      └────────────┬────────────┘ └────────────┬────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
               ┌───────────────────────────────────┐
               │    K Rho-Odds Components S_k(t)   │
               └─────────────────┬─────────────────┘
                                 ▼
               ┌───────────────────────────────────┐
               │ Overall Mixture Survival S(t|x)   │
               └───────────────────────────────────┘
```

1. **Representation Network ($\phi_\theta(\mathbf{x})$):** A deep neural network learning a dense feature embedding of raw covariates $\mathbf{x}$.
2. **Gating Network ($\pi_k(\mathbf{x})$):** Computes categorical assignment probabilities for $K$ latent mixture components via a softmax transformation, enforcing $\sum_{k=1}^{K} \pi_k(\mathbf{x}) = 1$.
3. **Rho-Odds Components ($S_k(t \mid \mathbf{x})$):** Latent survival formulations, each parameterized by component-specific regression weights $\beta_k$ and a baseline function $A_k(t)$.

### Mathematical Formulation

The marginal survival probability is the weighted mixture across all $K$ components:

$$S(t \mid \mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x}) S_k(t \mid \mathbf{x})$$

Where each individual latent component $S_k(t \mid \mathbf{x})$ is defined as:

$$S_k(t \mid \mathbf{x}) = \left[ 1 + \rho A_k(t) \exp\left( \beta_k^\top \phi_\theta(\mathbf{x}) \right) \right]^{-1/\rho}$$

---

## Theoretical Dynamics

### Mixture Hazard Function

The overall hazard rate $\lambda(t \mid \mathbf{x})$ is **not** a simple weighted average of component hazards. Conditioning on survival past time $t$ dynamicizes the effective component probabilities:

$$\lambda(t \mid \mathbf{x}) = \frac{\sum_{k=1}^{K} \pi_k(\mathbf{x}) S_k(t \mid \mathbf{x}) \lambda_k(t \mid \mathbf{x})}{\sum_{k=1}^{K} \pi_k(\mathbf{x}) S_k(t \mid \mathbf{x})}$$

The posterior component weight $P(Z = k \mid T > t, \mathbf{x})$ evolves over time:

$$P(Z = k \mid T > t, \mathbf{x}) = \frac{\pi_k(\mathbf{x}) S_k(t \mid \mathbf{x})}{\sum_{j=1}^{K} \pi_j(\mathbf{x}) S_j(t \mid \mathbf{x})}$$

> **Key Insight:** As $t$ increases, components with higher survival probabilities exert a stronger influence on the aggregate hazard curve.

### Generalized Proportional Rho-Odds Model

The baseline Proportional Rho-Odds model is parameterized as:

$$\log \left[ \frac{S(t \mid \mathbf{x})^{-\rho} - 1}{\rho} \right] = \alpha(t) + \beta^\top \mathbf{x}$$

$$\implies S(t \mid \mathbf{x}) = \left[ 1 + \rho \exp\{\alpha(t) + \beta^\top \mathbf{x}\} \right]^{-1/\rho}$$

The corresponding individual hazard function is given by:

$$\lambda(t \mid \mathbf{x}) = \frac{a(t) \exp(\beta^\top \mathbf{x})}{1 + \rho A(t) \exp(\beta^\top \mathbf{x})}$$

*where $A(t) = \exp\{\alpha(t)\}$ and $a(t) = A'(t)$.*

| Parameter Value | Equivalent Model |
| :--- | :--- |
| **$\rho \rightarrow 0$** | Proportional Hazards (Cox Model) |
| **$\rho = 1$** | Proportional Odds |

---

## Loss & Likelihood Optimization

For right-censored data consisting of tuples $(\mathbf{x}_i, u_i, \delta_i)$, where $u_i$ is the observed time and $\delta_i \in \{0, 1\}$ is the event indicator:

$$\mathcal{L} = \prod_{i=1}^{N} f(u_i \mid \mathbf{x}_i)^{\delta_i} S(u_i \mid \mathbf{x}_i)^{1 - \delta_i}$$

Model parameters are optimized by maximizing the log-likelihood function $\ell$:

$$\ell = \sum_{i=1}^{N} \left[ \delta_i \log \left( \sum_{k=1}^{K} \pi_k(\mathbf{x}_i) f_k(u_i \mid \mathbf{x}_i) \right) + (1 - \delta_i) \log \left( \sum_{k=1}^{K} \pi_k(\mathbf{x}_i) S_k(u_i \mid \mathbf{x}_i) \right) \right]$$

---

## Experimental Setup

### Hyperparameter Search Space

Experiments focus on evaluating parameter sensitivity across three core dimensions:

* **Mixture Components ($K$):** Optimal number of latent survival subpopulations.
* **Rho-Odds Parameter ($\rho$):** Impact of hazard structural constraints (interpolating between proportional hazards and proportional odds).
* **Baseline Functions ($A_k(t)$):** Evaluating parametric vs. flexible non-parametric formulations for time dependence.

### Benchmark Dataset

Experiments are conducted on the **SUPPORT** (*Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments*) dataset.

> 📌 **Data Availability:** The dataset is not distributed with this repository. Obtain `support.csv` from the official repository or public mirrors and place it into the `data/` directory prior to running experiments.

---

## Tech Stack & Dependencies

* **Core Libraries:** `Python 3.8+`
* **Deep Learning:** [PyTorch](https://pytorch.org/) (Model building, auto-differentiation)
* **Scientific Computing:** [SciPy](https://scipy.org/) & [NumPy](https://numpy.org/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/)
