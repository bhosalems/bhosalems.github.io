---
layout: post
title: KL divergence can not be Distance metric
date: 2024-11-28 19:45:00 +0530
description: KL divergence can ot be Distance metric
tags: kl-diveregence self-learning
categories: technical
pseudocode: true
math: true
---

### Distance metrics

In contrastive learning distance metric is used to bring the postive pairs closer and negative pairs farther apart. Distance metric therefore should be chosen carefully. In case of probability distributions we will investigate whether KL divergence is goos metric to use in the loss function. Distance metrics should MUST satify below properties

1. <span style="font-weight: bold;">Non-Negativity</span>: $$D(P, Q) >= 0$$.
2. <span style="font-weight: bold;">Identity of Indiscrenibles</span>: $$D(P, Q) = 0$$ iff $$P==Q$$.
3. <span style="font-weight: bold;">Symmetry</span>: $$D(P, Q) = D(Q, P)$$. Distance should not depend on the direction.
4. <span style="font-weight: bold;">Triangle Inequaltiy</span>: $$D(P, Q) <= D(P, R) + D(R, Q)$$. Direct distance is the shortest distance.

---

### KL Diveregence

The Kullback–Leibler (KL) divergence between two probability distributions is given by,

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

##### Tying with Information

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

{% include disqus.html %}
