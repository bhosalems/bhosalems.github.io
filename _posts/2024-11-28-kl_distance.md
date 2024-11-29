---
layout: post
title: Distance metric for fairness in vision laguage models
date: 2024-11-28 19:45:00 +0530
description: Distance metric for fairness
tags: kl-diveregence self-learning
categories: technical
pseudocode: true
math: true
---

### Section 1: Distance metrics

In contrastive learning distance metric is used to bring the postive pairs closer and negative pairs farther apart. Distance metric therefore should be chosen carefully. In case of probability distributions we will investigate whether KL divergence is goos metric to use in the loss function. Distance metrics should MUST satify below properties

1. <span style="font-weight: bold;">Non-Negativity</span>: $$D(P, Q) >= 0$$.
2. <span style="font-weight: bold;">Identity of Indiscrenibles</span>: $$D(P, Q) = 0$$ iff $$P==Q$$.
3. <span style="font-weight: bold;">Symmetry</span>: $$D(P, Q) = D(Q, P)$$. Distance should not depend on the direction.
4. <span style="font-weight: bold;">Triangle Inequaltiy</span>: $$D(P, Q) <= D(P, R) + D(R, Q)$$. Direct distance is the shortest distance.

---

### Section 2: KL Diveregence

The Kullback–Leibler (KL) divergence between two probability distributions is given by,

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

##### 2.1. Tying with Information

In information theory, <span style="font-weight: bold;">information</span> quantifies uncertainty reduction. For an event $$x$$ with probability $$P(x)$$, the information content is:

$$
I(x) = -\log P(x)
$$

Rare events contain more ifnormation than frequent events. The <span style="font-weight: bold;">entropy</span> of a distribution $$P$$ is the expected information content:

$$
H(P) = - \sum_x P(x) \log P(x)
$$

KL Divergence represents the additional information required to describe samples from $$P$$ using $$Q$$:

$$
D_{KL}(P \| Q) = \sum_x (P(x) \log P(x) - P(x) \log Q(x))
$$

- $$D_{KL}(P \| Q)$$ measures how poorly $$Q$$ approximates $$P$$.
- High divergence indicates significant <span style="font-weight: bold;">information mismatch</span>.

Therefore KL diveregence lacks the <span style="font-weight: bold;">geometric</span> intepretation of the distance and violates symmetric and triangle-inequlaity requirement i.e

$$
  D_{KL}(P \| Q) \neq D_{KL}(Q \| P)
$$

Example: If $$D_{KL}(P \| Q)$$ is large but $$D_{KL}(Q \| R)$$ is small, the sum $$D_{KL}(P \| Q) + D_{KL}(Q \| R)$$ might not meaningfully bound $$D_{KL}(P \| R)$$.

---
### Section 3: Sinkhorn Distance

The Sinkhorn Distance is a regularized variant of the Wasserstein Distance, designed for computational efficiency and smoothness. It is defined as:

$$
W_{\epsilon(D_B, D_{B_a})} = \inf_{\gamma \in \Gamma(D_B, D_{B_a})} \mathbb{E}_{(p, q) \sim \gamma}[c(p, q)] + \epsilon H(\gamma \| \mu \otimes \nu)
$$

Where:
- $$\\Gamma(D_B, D_{B_a})$$: The set of all joint distributions $$\gamma(p, q)$$ such that:
$$
  \int \gamma(p, q) dq = D_B(p) \quad \text{first marginal matches } D_B, 
$$
$$
  \int \gamma(p, q) dp = D_{B_a}(q) \quad \text{second marginal matches } D_{B_a}.
$$

- $$c(p, q)$$: The transport cost (e.g., Euclidean distance) between points $$p$$ and $$q$$.
- $$\epsilon$$: A regularization parameter controlling the weight of the entropy term.
- $$H(\gamma \| \mu \otimes \nu)$$: The relative entropy (or KL Divergence) between $$\gamma$$ and the product measure $$\mu \otimes \nu$$.

##### 3.1. Key Components

###### Joint Distribution $$\gamma$$
- $$\gamma(p, q)$$ represents a joint probability distribution over $$p$$ and $$q$$, where:
  - $$p$$ is sampled from $$D_B$$ (general batch distribution).
  - $$q$$ is sampled from $$D_{B_a}$$ (subgroup-specific distribution).
- **Marginal Constraints**: 
  - The first marginal of $$\gamma$$ is $$D_B$$.
  - The second marginal of $$\gamma$$ is $$D_{B_a}$$.

###### **Transport Cost $$\mathbb{E}_{(p, q) \sim \gamma}[c(p, q)]$$**
- This term minimizes the cost of moving mass between $$D_B$$ and $$D_{B_a}$$.
- Alignment is achieved by finding an optimal $$\\gamma$$ that connects $$D_B$$ and $$D_{B_a}$$ while minimizing this cost.

###### **Regularization Term $$H(\gamma \| \mu \otimes \nu)$$**
- $$\\mu \otimes \nu$$ is a general measure, not necessarily a probability measure.
- Regularization encourages $$\gamma$$ to have a smoother distribution, influenced by $$\mu \otimes \nu$$.

---

##### 3.2. Role of $$\gamma$$ in Fairness Learning

In fairness learning, the Sinkhorn Distance aligns the distributions $$D_B$$ (general) and $$D_{B_a}$$ (specific to a subgroup). The joint distribution $$\gamma$$:

1. **Ensures Marginal Consistency**:
   - $$ \int \gamma(p, q) dq = D_B(p) $$.
   - $$ \int \gamma(p, q) dp = D_{B_a}(q) $$.

2. **Balances Objectives**:
   - Align $$D_B$$ and $$D_{B_a}$$ via transport cost minimization.
   - Smooth the distribution $$\gamma$$ via entropy regularization.

3. **Optimal Transport Plan**:
   - $$\gamma(p, q)$$ determines how much "mass" is moved from $$p$$ (in $$D_B$$) to $$q$$ (in $$D_{B_a}$$).

---

##### 3.3. Why Sinkhorn Distance?

Compared to KL Divergence:
1. **Handles Marginal Constraints**: 
   - Ensures the distributions $$D_B$$ and $$D_{B_a}$$ are matched via the transport plan $$\gamma$$.
2. **Incorporates Regularization**:
   - The term $$H(\gamma \| \mu \otimes \nu)$$ smooths $$\gamma$$, making computation efficient and avoiding degenerate solutions.
3. **True Metric**:
   - Unlike KL Divergence, Sinkhorn Distance satisfies the properties of a true metric (e.g., triangle inequality).

---

{% include disqus.html %}
