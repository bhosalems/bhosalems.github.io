---
layout: post
title: Distance metric for fairness in vision language models
date: 2024-11-28 19:45:00 +0530
description: Distance metric for fairness
tags: kl-diveregence self-learning
categories: technical
pseudocode: true
math: true
---

### Section 1: Distance metrics

In contrastive learning distance metric is used to bring the postive pairs closer and push negative pairs farther apart. Distance metric therefore should be chosen carefully. In case of probability distributions we will investigate whether KL divergence is good metric to use in the loss function. Distance metrics <span style="font-weight: bold;">MUST</span> satify below properties,

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

Rare events contain more information than frequent events. The <span style="font-weight: bold;">entropy</span> of a distribution $$P$$ is the expected information content:

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

and for example: If $$D_{KL}(P \| Q)$$ is large but $$D_{KL}(Q \| R)$$ is small, the sum $$D_{KL}(P \| Q) + D_{KL}(Q \| R)$$ might not meaningfully bound $$D_{KL}(P \| R)$$.

---
### Section 3: Fairness Background {% cite luo2024fairclip --file 2024-11-28-kl_distance %}

With the labeled data $$\mathcal{D} = \{(x^I_i, x^T_i, y_i, a_i)\}$$, where $$x^I_i \in \mathcal{X}^I$$ is an image, $$x^T_i \in \mathcal{X}^T$$ is a corresponding text, $$y_i \in \mathcal{Y}$$ is a label (e.g., malignant tumor or benign tumor), and $$a_i \in \mathcal{A}$$ is an identity attribute (e.g., gender, race, ethnicity), we employ a vision-language (VL) pre-trained model (e.g., CLIP) $$f$$ with a vision encoder $$f_I$$ and a text encoder $$f_T$$ to generate vision features $$z_I$$ and text features $$z_T$$.

Given a batch of $$N$$ image-text pairs, CLIP produces a similarity matrix $$M \in \mathbb{R}^{N \times N}$$ with $$M_{ij} = z_{I_i}^\top z_{T_j}$$, where each entry represents the cosine similarity between the $$i$$-th image feature and the $$j$$-th text feature. CLIP optimizes a contrastive loss to maximize similarity for the $$N$$ positive pairs and minimize it for the $$N^2 - N$$ negative pairs:

$$
\min_f - \sum_{i=1}^N \sum_{j=1}^N \delta(i-j) \log\left(\frac{z_{I_i}^\top z_{T_j}}{\|z_{I_i}\|\cdot\|z_{T_j}\|}\right),
$$

where $$\delta(i-j)=1$$ if $$i=j$$ and $$0$$ otherwise.

For fairness, the model $$f \in \mathcal{F}: \mathcal{X}^I \times \mathcal{X}^T \times \mathcal{A} \xrightarrow[]{\theta} \mathcal{Y}$$ incorporates identity attributes into training. Fairness learning thus seeks to minimize disparities between identity groups while also maintaining high accuracy. Fairness during the pre-training phase can thus be improved by minimizing the disparity between the distributions of $$M$$ the correlation between vision and language features—across different identity groups.

Let $$D_{\{(x^I, x^T, a)\}\mid f}$$ denote the distribution of $$M_{i,i}$$ under model $$f$$, and $$D_{\{(x^I, x^T, a) \mid a=\alpha\}\mid f}$$ be the corresponding distribution restricted to group $$\alpha$$. The fairness objective is:

$$
\min_f \sum_{\alpha \in A} d\bigl(D_{\{(x^I, x^T, a)\}|f}, D_{\{(x^I, x^T, a)|a=\alpha\}|f}\bigr),
$$

where $$d$$ is a distance function. Since these true distributions are intractable, we instead use empirical distributions $$D_{B \mid f}$$ and $$D_{B_a \mid f}$$ estimated from a batch $$B$$ and its subgroup batch $$B_a$$. Concretely,
$$
D_B  = \sum_{i=1}^{\mid B \mid} p_i \delta_{x_i}
$$
and
$$
D_{B_a}  = \sum_{i=1}^{\mid B_a \mid} q_i \delta_{y_i}
$$
where,
  - $$x_i$$ are data points in $$D_B$$
  - $$y_j$$ are data points in $$D_{B_a}$$
  - $$p_i$$ and $$q_j$$ are probability weights assigned to each data points $$x_i$$ and $$y_j$$
  - $$\delta_{x_i}$$ and $$\delta_{y_j}$$ are dirac delta functions centered at $$x_i$$ and $$y_j$$ i.e.

    $$
      \delta_i(x) =
      \begin{cases} 
      1, & \text{if } x = x_i, \\
      0, & \text{otherwise.}
      \end{cases}
    $$
    and
    $$
      \delta_j(y) =
      \begin{cases} 
      1, & \text{if } y = y_j, \\
      0, & \text{otherwise.}
      \end{cases}
    $$

---

### Section 4: Sinkhorn Distance
The Sinkhorn Distance is a regularized variant of the Wasserstein Distance, designed for computational efficiency and smoothness. It is defined as:

$$
W_{\epsilon(D_B, D_{B_a})} = \inf_{\gamma \in \Gamma(D_B, D_{B_a})} \mathbb{E}_{(x_i, y_j) \sim \gamma}[c(x_i, y_j)] + \epsilon H(\gamma \| \mu \otimes \nu)
$$

Where:

- $$\Gamma(D_B, D_{B_a})$$: The set of all joint distributions $$\gamma(x_i, y_j)$$ such that:

  $$
    \sum_{j=1}^{\mid D_{B_a}\mid} \gamma(x_i, y_j)  = p_i \quad \text{first marginal matches } D_B,
  $$

  $$
    \sum_{i=1}^{\mid D_{B}\mid} \gamma(x_i, y_j) = q_j \quad \text{second marginal matches } D_{B_a}.
  $$

- $$c(x_i, y_j)$$: The transport cost (e.g., Euclidean distance) between points $$x_i$$ and $$y_j$$.
- $$\epsilon$$: A regularization parameter controlling the weight of the entropy term.
- $$H(\gamma \| \mu \otimes \nu)$$: The relative entropy between $$\gamma$$ and the product measure $$\mu \otimes \nu$$.

---

##### 4.1. Key Components

###### Joint Distribution $$\gamma$$

- $$\gamma(x_i, y_j)$$ represents a joint probability distribution over $$i$$ and $$j$$, where:
  - $$x_i$$ is sampled from $$D_B$$ (general batch distribution).
  - $$y_j$$ is sampled from $$D_{B_a}$$ (subgroup-specific distribution).
- **Marginal Constraints**: As stated earlier, this is important.

###### Transport Cost $$\mathbb{E}_{(x_i, y_j) \sim \gamma}[c(x_i, y_j)]$$

- This term minimizes the cost of moving mass between $$D_B$$ and $$D_{B_a}$$.
- Alignment is achieved by finding an optimal $$\gamma$$ that connects $$D_B$$ and $$D_{B_a}$$ while minimizing this cost.

###### Regularization Term $$H(\gamma \| \mu \otimes \nu)$$

- $$\mu \otimes \nu$$ is a general measure, not necessarily a probability measure.
- Regularization encourages $$\gamma$$ to have a smoother distribution, influenced by $$\mu \otimes \nu$$.

---

##### 4.2. Role of $$\gamma$$ in Fairness Learning

In fairness learning, the Sinkhorn Distance aligns the distributions $$D_B$$ (general) and $$D_{B_a}$$ (specific to a subgroup). The joint distribution $$\gamma$$:

1. **Ensures Marginal Consistency**:

   - $$\sum_{j=1}^{\mid D_{B_a}\mid} \gamma(x_i, y_j)=p_i$$ and  $$\sum_{i=1}^{\mid D_{B}\mid} \gamma(x_i, y_j)=q_j$$

2. **Balances Objectives**:

   - Align $$D_B$$ and $$D_{B_a}$$ via transport cost minimization.
   - Smooth the distribution $$\gamma$$ via entropy regularization.

3. **Optimal Transport Plan**:
   - $$\gamma(x_i, y_j)$$ determines how much "mass" is moved from $$x_i$$ (in $$D_B$$) to $$y_j$$ (in $$D_{B_a}$$).

---

##### 4.3. Why Sinkhorn Distance?

Compared to KL Divergence:

1. **Handles Marginal Constraints**:
   - Ensures the distributions $$D_B$$ and $$D_{B_a}$$ are matched via the transport plan $$\gamma$$.
2. **Incorporates Regularization**:
   - The term $$H(\gamma \| \mu \otimes \nu)$$ smooths $$\gamma$$, making computation efficient and avoiding degenerate solutions.
3. **True Metric**:
   - Unlike KL Divergence, Sinkhorn Distance satisfies all the properties of a true metric (e.g., triangle inequality).

---
## References:<br/>

{% bibliography --file 2024-11-28-kl_distance %}

<br/>
{% include disqus.html %}
