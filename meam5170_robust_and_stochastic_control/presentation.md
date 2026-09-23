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
    \setbeamersize{text margin left=0.5cm,text margin right=0.5cm}
    \setbeamertemplate{frametitle}{\nointerlineskip\vspace*{0.5cm}\begin{beamercolorbox}[leftskip=0pt,rightskip=0pt]{frametitle}\usebeamerfont{frametitle}\insertframetitle\par\end{beamercolorbox}\vspace*{0cm}}
    \setbeamertemplate{footline}[page number]
    \setbeamerfont{page number in head/foot}{size=\normalsize}
    \setbeamertemplate{caption}{}
    \setlength{\abovecaptionskip}{0pt}
    \setlength{\belowcaptionskip}{0pt}
    \renewcommand{\caption}[2][]{ }
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

# Continuous vs Discrete

We can talk about stochastic systems in either continous or discrete time. For simplicity, this lecture mostly focuses on discrete time. 

In continuous time, you deal with **stochastic differential equations**, and have to worry about Itô's lemma, which describes how the chain rule works for SDE's.

*For example,* if $W_t$ is brownian noise, it follows from Itô's lemma that:
$$ \frac{d}{dt}\Big[W_t^2\Big] = 2 W_t \frac{d W_t}{dt} + 1 $$

<!-- 
NOTE: this is actually incorrect, there is really not a dW_t / dt. Instead, the equation should be d[W_t^2] = 2 W_t dW_t + dt.
-->



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
$$ x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) \text{ I.I.D.} $$

<!-- 
ANSWER: x_t \to N(0, P), where P satisfies P = A_* P A_*^T + \Sigma.
-->

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

# Observability

Our current formulation, 
$$ x_{t+1} = f(x_t, u_t, w_t) $$ 
doesn't give us uncertainty about state from partial-observability. This requires introducing observations:
$$ y_t = h(x_t, u_t, v_t) $$
where $v_t$ is a stochastic process handling measurement noise.

**Note:** *We now have a partially-observable markov decision process.*

# Robust vs Stochastic Control

$$ x_{t+1} = f(x_t, u_t, w_t), \qquad x_0 \sim p(x_0), $$

Let $J(\pi; w_{0:T-1})$ be a cost function over a policy based on $w_{0:T-1}$. 

Let $\pi \in \Pi$ be policy, which in the case of trajectory optimization may be parameterized by the individual control inputs $u_0, u_1, ..., u_{T-1}$.

In general, **stochastic control** aims to optimize the expected objective:
$$ \min_{\pi \in \Pi} \; \mathbb E_w J(\pi; w_{0:T-1}) $$
Whereas **robust control** aims to optimize the worst-case objective:
$$ \min_{\pi \in \Pi} \; \max_{w_{0:T-1} \in W} J(\pi; w_{0:T-1}) $$

# Robust and Stochastic Value Functions

We can think about value functions for a running cost $c(x, u, w)$

**Stochastic:**
$$ V(x) = \min_u \mathbb E_w \left[ c(x, u, w) + \gamma V(f(x, u, w)) \right] $$

**Robust:**
$$ V(x) = \min_u \max_w \left[ c(x, u, w) + \gamma V(f(x, u, w)) \right] $$

**Question:** *what is the problem with these formulations?*

<!-- 
ANSWER: you need to make assumptions for them to be valid:

- for the stochastic control value function, you need w_t to be constant over time and only depend on the current x, u (not prev observations).
- for the robust control value function you need rectangularity of W

 -->

# Example: Domain Randomization in RL

Reinforcement learning with domain randomization tries to minimize some cost:
$$ J(\theta) = \mathbb E_\xi \mathbb E_{\tau \sim \pi_\theta, \xi} \left[ \sum_{t=1}^T \gamma^t c(x, u) \right] $$
where $\xi$ are random domains (friction, etc.)

Interestingly, domain randomization can some times result in a better policy on the nominal dynamics than not doing domain randomization:

TODO: image


# Example: Stochastic LQR

$$ x_{t+1} = A x_t + B u_t + w_t, \qquad w_t \sim \mathcal N(0, \Sigma) $$
Consider the cost $c(x, u) = x^\top Q x + u^\top R u$ for $Q = Q^\top \succeq 0$, $R = R^\top \succ 0$. Then, our value function is:
$$ V(x) = \min_u \mathbb E_w \left[ x^\top Q x + u^\top R u + \gamma V(Ax + Bu + w) \right] $$
Let's assume our optimal policy is $u_t = K x_t$ and our value function is $V(x) = x^\top S x + c$ for some $S = S^\top \succ 0$ and $c$. Then, we get:
$$ S = Q + \gamma A^\top S A - \gamma^2 A^\top S B(R + \gamma B^\top S B)^{-1} B^\top S A $$
$$ c = \frac{\gamma}{1-\gamma} \mathbb E_w [w^\top S w] = \frac{\gamma}{1-\gamma} \text{tr}(S \Sigma) $$

**Question:** *what happens if $\gamma=1$?*

<!-- # Margins of Stochastic LQR

**Note:** *Stochastic LQR inherits the same same margins as LQR*

However, if a Kalman filter is in the loop, those margins go away  -->

# Example: LQG

TODO: this


# Example: Tube Trajectory Optimization

TODO: this


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

# Chance Constraints

TODO: this

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



