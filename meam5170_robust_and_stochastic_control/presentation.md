---
title: "Robust & Stochastic Control"
subtitle: "MEAM 5170"
# date: 2026 September 29
# author: Herbie Wright
theme: default
colortheme: seagull
fonttheme: professionalfonts
classoption: t
mainfont: "Roboto"
sansfont: "EB Garamond"
fontsize: 10pt
header-includes:
  - |
    ```{=latex}
    \setlength{\parskip}{1em}
    \setbeamerfont{frametitle}{size=\large}
    \setbeamerfont{structure}{family=\sffamily}
    \setbeamersize{text margin left=1cm,text margin right=1cm}
    \setbeamertemplate{frametitle}{\nointerlineskip\vspace*{0.5cm}\begin{beamercolorbox}[leftskip=0pt,rightskip=0pt]{frametitle}\usebeamerfont{frametitle}\insertframetitle\par\end{beamercolorbox}\vspace*{0cm}}
    \setbeamertemplate{footline}[page number]
    \setbeamerfont{page number in head/foot}{size=\normalsize}
    \setbeamertemplate{caption}{}
    \setlength{\abovecaptionskip}{0pt}
    \setlength{\belowcaptionskip}{0pt}
    \renewcommand{\caption}[2][]{}
    \setbeamertemplate{itemize item}{\textbullet}
    ```
---

# Last Time: Trajectory Optimization

Here is a general form of discrete-time trajectory optimization:

\begin{align*}
    \min_{x_{0:T}, u_{0:T-1}} &\; c_T(x_T) + \sum_{t=0}^{T-1} c(x_t, u_t) \\
    \text{s.t.} &\; x_{t+1} = f(x_t, u_t), \quad x_0 = x_\text{init},
\end{align*}

# A Quadrotor Example

Say we want to find a minimum length trajectory for a quadrotor to get to point A to point B:

![.](image.png){width=50%}

**Question:** *what could go wrong?*

# Defining a Stochastic Control System

Previously:
$$ x_{t+1} = f(x_t, u_t), \qquad x_0 = x_\text{init} $$

Now:
$$ x_{t+1} = f(x_t, u_t, w_t), \qquad x_0 \sim p(x_0), $$
where $w_t$ is a random process.

<!-- 
NOTE: can model things like disturbances, parameter uncertainty, etc
Also, w_t don't need to be independent.
-->

# Continuous vs Discrete

We can talk about stochastic systems in either continous or discrete time. For simplicity, this lecture mostly focuses on discrete time. 

In continuous time, you deal with **stochastic differential equations**, and have to worry about Itô's lemma, which describes how the chain rule works for SDE's.

<!-- *For example,* if $W_t$ is brownian noise, it follows from Itô's lemma that:
$$ d\Big[W_t^2\Big] = 2 W_t d W_t + dt $$ -->

# Example: Stochastic LTI

Consider:
$$ x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) $$
For some noise covariance $\Sigma = \Sigma^\top \succ 0$

**Note:** *if $x_0$ is a Gaussian and $u_t$ is an affine function of $x_t$, then all $x_t$ are Gaussian.*

Let $u_t = K x_t$ such that $\rho(A + BK) < 1$.

**Question:** Does the system have a fixed point at $x = 0$?

<!-- 
ANSWER: no not really
 -->

# Stationary Distributions

For stochastic systems, it often makes more sense to talk about a *stationary distribution* as opposed to a stationary point.

![.](./slti.png){width=80%}

**Question:** *For the STLI system, if $x_0 = 0$, what happens to $x_t$ as $t \to \infty$?*
$$ x_{t+1} = (A + BK) x_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) \text{ I.I.D.} $$

<!-- 
ANSWER: x_t \to N(0, P), where P satisfies P = A_* P A_*^T + \Sigma.
-->

# Example: Polytope Uncertainty and LTI

Consider:
$$ x_{t+1} = (A + BK) x_t + w_t, \qquad w_t \in \mathcal W, $$
where $\mathcal W \subset \mathbb R^n$ is a convex polytope and $x_0 = 0$.

The *Minkowski sum* of two sets is 
$$ \mathcal X \oplus \mathcal W = \{x + w: x \in \mathcal X \; w \in \mathcal W\} $$

Also define $A \mathcal X = \{Ax: x \in \mathcal X\}$ for matrix $A$ and set $\mathcal X$.

Then our set-valued dynamics are:
$$ \mathcal X_{t+1} = (A + BK) \mathcal X_t \oplus \mathcal W $$
We can track the convex polytope $\mathcal X_{t}$ by its vertices $\{v_x^{(1)}, ..., v_x^{(m)}\}$

<!-- For $t=1, ..., T$:

1. Set empty vertex set $X_t \gets \emptyset$
2. For each $x \in X_{t-1}$ and $w \in W$, add $x + w$ to $X_t$
3. filter out points in the interior of $\text{Hull}(X_t)$. -->

# Zonotopes

Things are easier if our sets are *Zonotopes*, defined:
$$ \mathcal Z(c, G) = \left\{c + \sum_{i=1}^m z_i g_i: z_i \in [-1, 1]\right\}$$
Then, the Minkowski sum becomes:
$$ \mathcal Z(c, G) \oplus \mathcal Z(c', G') = \mathcal Z(c + c', [G,\;G']) $$
So, if $\mathcal X_t = \mathcal Z(c_x, G_x)$ and $\mathcal W = \mathcal Z(c_w, G_w)$, we get:
$$ \mathcal X_{t+1} = \mathcal Z\Big((A + BK) c_x + c_w, \big[(A + BK)G_x,\;G_w\big]\Big) $$

# Example: Zonotope evolution

![.](./zlti.png){width=80%}

**Takeaway:** *We can think about propogating sets through dynamics to model uncertainty*


# Example: Manipulator Equations with Uncertainty

Recall:
$$ M(q)\ddot q + C(q, \dot q)\dot q + g(q) = B u $$

We might have uncertainty about physical parameters (e.g. mass, friction). We can represent those by $\xi$:
$$ M(q, \xi)\ddot q + C(q, \dot q, \xi)\dot q + g(q, \xi) = B u $$
Then we would get the system:
$$ \begin{bmatrix} \dot q \\ \ddot q \end{bmatrix} = \begin{bmatrix} \dot q \\ M(q, \xi)^{-1} \left( B u - C(q, \dot q, \xi) \dot q - g(q, \xi)\right) \end{bmatrix}$$
where our random process, $w_t = \xi$ is a random constant (over time).

# Disturbances vs Uncertainty about Parameters

Stochastic system:
$$ x_{t+1} = f(x_t, u_t, w_t), \qquad x_0 \sim p(x_0), $$
We might want to separate random parameters $\xi$ that remain constant over time:
$$ x_{t+1} = f(x_t, u_t, w_t, \xi), \qquad x_0 \sim p(x_0) $$
Where $\xi$ is a random variable

**Note:** *Random parameters can be wrapped into the state by creating an augmented state (but then your controller loses full observability)* 


# Robust vs Stochastic Control

$$ x_{t+1} = f(x_t, u_t, w_t), \qquad x_0 \sim p(x_0), $$

Let $J(\pi; w_{0:T-1})$ be a cost function over a policy based on $w_{0:T-1}$. 

Let $\pi \in \Pi$ be policy, which in the case of trajectory optimization may be parameterized by the individual control inputs $u_0, u_1, ..., u_{T-1}$.

In general, **stochastic control** aims to optimize the expected objective:
$$ \min_{\pi \in \Pi} \; \mathbb E_w J(\pi; w_{0:T-1}) $$
Whereas **robust control** aims to optimize the worst-case objective:
$$ \min_{\pi \in \Pi} \; \max_{w_{0:T-1} \in W} J(\pi; w_{0:T-1}) $$

# Ex: Stoch. Trajectory Optimization via Sampling

Consider a stochastic trajectory optimization problem:
\begin{align*}
    \min_{X_{0:T}, u_{0:T-1}} &\; \mathbb E \left[ c_T(X_T) + \sum_{t=0}^{T-1} c(X_t, u_t) \right] \\
    \text{s.t.} &\; X_{t+1} = f(X_t, u_t, W_t), \quad X_0 = x_\text{init},
\end{align*}
an easy way to approximate this is to sample $\left\{w^{(i)}_t\right\}$ and solve:
\begin{align*}
    \min_{x_{0:T}^{(i)}, u_{0:T-1}} &\; \frac{1}{n}\sum_{i=1}^n \left[ c_T(x_T^{(i)}) + \sum_{t=0}^{T-1} c(x_t^{(i)}, u_t) \right] \\
    \text{s.t.} &\; x^{(i)}_{t+1} = f(x_t^{(i)}, u_t, w_t^{(i)}), \quad x_0^{(i)} = x_\text{init}
\end{align*}

**Question:** *why might sharing the $u_t$ between particles be too conservative?*

<!-- **Question:** *does this work for the robust version?* -->


# Example: Robust TO with Finite Uncertainty Set

Let's presume our uncertainty is drawn from a finite set, $\{ \xi^{(1)}, ..., \xi^{(n)} \}$, then we can do a similar thing:
\begin{align*}
    \min_{x_{0:T}^{(i)}, u_{0:T-1}} &\; \max_{i \in 1, ..., n} \left\{ c_T(x_T^{(i)}) + \sum_{t=0}^{T-1} c(x_t^{(i)}, u_t) \right\} \\
    \text{s.t.} &\; x^{(i)}_{t+1} = f(x_t^{(i)}, u_t, \xi^{(i)}), \quad x_0^{(i)} = x_\text{init}
\end{align*}

**Question:** *why do we have to assume the uncertainty set is finite for this? Why can't we sample?*

# Example: Tube Trajectory Optimization

Let's go back to a bounded uncertainty set $\mathcal W$, and consider linearized dynamics around a nominal trajectory and error feedback.

We want to optimize a *tube* around it so that we are guaranteed to remain in the tube. We pick an invariant set $\mathcal E$ and can formulate:
\begin{align*}
    \min_{\bar x_{0:T}, \bar u_{0:T-1}, \alpha_{0:T}} &\; c_T(\bar x_T^{(i)}) + \sum_{t=0}^{T-1} c(\bar x_t, \bar u_t) + \sum_{t=1}^T \gamma \alpha_t \\
    \text{s.t.} &\; \bar x_{t+1} = A_t \bar x_t + B_t \bar u_t \quad \bar x_0 = x_\text{init} \\
    &\; \alpha_{t+1} \mathcal E \supseteq (A_t + B_t K_t) (\alpha \mathcal E) \oplus \mathcal W \quad \alpha_t \geq 0
\end{align*}

If $\mathcal E = \{x \in \mathbb R: \|x\| \leq 1\}$ and $\mathcal W$ has radius $r_w$, the tube dynamics constraint becomes:
$$ \alpha_{t+1} \geq \| A + B K \|_2 \alpha + r_w $$


# Example: Tube Trajectory Optimization

TODO: image

# Chance Constraints

We don't always just care about the objective; we may also want *constraints* to be satisfied (think about the quadrotor).

However, a hard constraint like:
$$ g(x_t, u_t) \leq 0 $$
might not be feasible in stochastic systems (e.g. $x_t$ has Gaussian noise injected)

Instead, we can express a *chance constraint* as:
$$ P\left[g(x_t, u_t) \leq 0\right] \geq 1-\alpha $$

# How to Actually Solve Chance Constraints

TODO: this

- sampling
- propogate a parametric distribution (e.g. Gaussian) and put quantile constraints
- Show example of optimization problem

# Stochastic Value Function

- state assumptions
- derive the bellman equation

# Robust Value Function

- state rectangularity
- derive the bellman equation

<!-- 
# Robust and Stochastic Value Functions

We can think about value functions for a running cost $c(x, u, w)$

**Stochastic:**
$$ V(x) = \min_u \mathbb E_w \left[ c(x, u, w) + \gamma V(f(x, u, w)) \right] $$

**Robust:**
$$ V(x) = \min_u \max_w \left[ c(x, u, w) + \gamma V(f(x, u, w)) \right] $$

**Question:** *when are these not valid?*

<!-- 
ANSWER: you need to make assumptions for them to be valid:

- for the stochastic control value function, you need w_t to be constant over time and only depend on the current x, u (not prev observations).
- for the robust control value function you need rectangularity of W

 --> 




# Example: Stochastic LQR

$$ x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) $$
Consider the cost $c(x, u) = x^\top Q x + u^\top R u$ for $Q = Q^\top \succeq 0$, $R = R^\top \succ 0$. Then, our value function is:
$$ V(x) = \min_u \mathbb E_w \left[ x^\top Q x + u^\top R u + \gamma V(Ax + Bu + w) \right] $$
Let's assume our optimal policy is $u_t = K x_t$ and our value function is $V(x) = x^\top S x + c$ for some $S = S^\top \succ 0$ and $c$. Then, we get:
$$ S = Q + \gamma A^\top S A - \gamma^2 A^\top S B(R + \gamma B^\top S B)^{-1} B^\top S A $$
$$ c = \frac{\gamma}{1-\gamma} \mathbb E_w [w^\top S w] = \frac{\gamma}{1-\gamma} \text{tr}(S \Sigma) $$

**Question:** *what happens if $\gamma=1$?*

# When $\gamma=1$

TODO: this

<!-- # Margins of Stochastic LQR

**Note:** *Stochastic LQR inherits the same same margins as LQR*

However, if a Kalman filter is in the loop, those margins go away  -->

# Observability

Our current formulation, 
$$ x_{t+1} = f(x_t, u_t, w_t) $$ 
doesn't give us uncertainty about state from partial-observability. This requires introducing observations:
$$ y_t = h(x_t, u_t, v_t) $$
where $v_t$ is a stochastic process handling measurement noise.

<!-- **Note:** *We can now model a partially-observable markov decision processes.*

# Aside: Partially Observable Markov Decision Processes

A POMDP is a tuple $(\mathcal S, \mathcal A, \mathcal O, T, O, R, \gamma)$ where:

|Symbol| Meaning|
| - | - |
| $\mathcal S$ | the set of states |
| $\mathcal A$ | the set of actions |
| $\mathcal O$ | the set of observations |
| $T(s_{t+1} | s_t, a_t)$ | the distribution over state transitions |
| $O(o_{t+1} | s_{t+1}, a_t)$ |the distribution over observations |
| $R(s_t, a_t)$ | the reward model |
| $\gamma \in [0, 1]$| the discount rate | -->

# Example: LQG

$$ x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) $$
We can extend our Stochastic LTI system to have uncertain observations:
$$ y_t = C x_t + v_t, \qquad v_t \sim \mathcal N (0, \Omega) $$
Then, we can arrive at a *Kalman filter*. 

Adding a Kalman filter in the loop with LQR gives us LQG: Linear-Quadratic-Gaussian


# $H_\infty$ Control

Consider a system with disturbances $w(t)$ and cost-relevant outputs $z(t)$:
$$ \dot x(t) = f_\pi (x(t), w(t)), \qquad z(t) = h_\pi (x(t), w(t)) $$
where $f_\pi, h_\pi$ depend on the chosen policy $\pi$.

Define a norm over a time-varying function:
$$ \left\| x(\cdot) \right\|_{L_2} = \sqrt{\int_0^T x(t)^\top x(t) dt} $$

The $H_\infty$ problem is then:
$$ \min_{\pi \in \Pi} \; \max_{\|w\|_{L_2} \neq 0} \; \frac{\|z(\cdot)\|_{L_2}}{\|w(\cdot)\|_{L_2}} $$

# Lyapunov Functions as Certificates in $H_\infty$ Control

Suppose we want to build a certificate of our objective:
$$ \max_{\|w\|_{L_2} \neq 0} \frac{\|z(\cdot)\|_{L_2}}{\|w(\cdot)\|_{L_2}} \le \gamma. $$
This is equivalent to saying the following holds for all $\|w\|_{L_2}$:
$$ \|z(\cdot)\|_{L_2}^2 - \gamma^2 \|w(\cdot)\|_{L_2}^2 \le 0. $$

We can actually certify this by finding a Lyapunov function $V(x)$ satisfying:
$$ \dot V(x) + z(t)^\top z(t) - \gamma^2 w(t)^\top w(t) \le 0. $$
For all $w(\cdot)$.


# Example: Domain Randomization in RL

Reinforcement learning with domain randomization tries to minimize some cost:
$$ J(\theta) = \mathbb E_\xi \mathbb E_{\tau \sim \pi_\theta, \xi} \left[ \sum_{t=1}^T \gamma^t c(x, u) \right] $$
where $\xi$ are random domains (friction, etc.)

Interestingly, domain randomization can some times result in a better policy on the nominal dynamics than not doing domain randomization:

TODO: image

# Other Measures of Risk: VaR and CVaR

Let $J$ be a random variable denoting some sort of cost (e.g. the trajectory cost over a random $w_t$). Maybe we don't want to just focus on $\mathbb E J$.

The value-at-risk at level $\alpha$ is a quantile:
$$ \operatorname{VaR}_\alpha(J) = \min \{ z : \mathbb P(J \le z) \ge \alpha \} $$
The conditional value-at-risk is the expected tail cost:
$$ \operatorname{CVaR}_\alpha(J) = \mathbb E[J \mid J \ge \operatorname{VaR}_\alpha(J)] $$

TODO: image

# (Manipulation) Funnels

TODO: this

# Distributionally Robust Control

TODO: this


