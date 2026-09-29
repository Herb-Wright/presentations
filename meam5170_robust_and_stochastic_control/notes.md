---
title: Lecture notes (2026 Sept. 28)
---

### Slide 1

- Fall break, so no class on Thursday
- No office hours this week
- Homework 5 released: due on October 7

### Slide 2

- We can add in constraints for $x$ and $u$
- If the cost is convex-quadratic and dynamics/constraints are linear, it is a QP

### Slide 3

- there might be wind (disturbance)
- we might misjudge our position (state uncertainty)
- we might misjudge where the obstacle is (parameter uncertainty)

### Slide 4

- this formulation is general, $w_t$ can be parameters, disturbances, everything in between
- $x_t$ are now all random variables

### Slide 5

- the math is a bit more complex, but fun
- Black-Scholes equation

### Slide 6

- If the were no disturbances there would be a fixed point
- There is no fixed point

### Slide 7

- the stability pushes in, the disturbance pushes out
- We know $x_t$ are Gaussian, so we can consider the mean and covariance

The mean:
$$  \mathbb E x_{t+1} = \mathbb E[A_K x_t] + \mathbb E[w_t] $$ 
$$  = A_K \mathbb E x_t $$ 
$$  = A_K^t \mathbb E x_0 = 0 $$ 

The covariance:
$$  \mathbb [x_{t+1} x_{t+1}^\top] = \mathbb E[A_K x_t x_t^\top A_K^\top] + \mathbb E[A_K x_t w_t^\top] + \mathbb E [w_t x_t^\top A_k^\top] + \mathbb E[w_t w_t^\top] $$ 
$$  = A_K \mathbb [x_t x_t^\top] A_K^\top + \Sigma = \sum_{i=0}^{t-1} A_K \Sigma A_K^\top $$ 
which converges to some $\Gamma \succ 0$ because $\rho(A_K) < 1$. Thus,
$$  x_t \xrightarrow{d} \mathcal N(0, \Gamma) $$ 

### Slide 8

- Algorithm: loop through vertices in $\mathcal W$ and vertices in $\mathcal X_t$ and make a new set of vertices
- Problem: $O(N^T)$ vertices

### Slide 9

- Now, it only takes $O(N T)$ space.

### Slide 10

- there is a stationary set here, similar to the SLTI, but it need not be a finite zonotope

### Slide 11

### Slide 12

Augmented state formulation:
$$  \begin{bmatrix}x_{t+1} \\ \xi_{t+1}\end{bmatrix} = \begin{bmatrix} f(x_t, u_t, w_t, \xi) \\ \xi_{t} \end{bmatrix}, \qquad \begin{bmatrix}x_0 \\ \xi_0\end{bmatrix} \sim \begin{bmatrix} p(x_0) \\ p(\xi) \end{bmatrix} $$ 

Note: you may lose observability

### Slide 13

- Note that $J$ is now a random variable because of stochasticity
- There are things in between average and worst case

### Slide 14

- it is hard to do optimization where things are random variables
- sampling is drop-dead easy; but how many do you need?
- does feedback have a role in trajectory optimization? Does it change the number of samples?
- feedback makes the problem non-convex

### Slide 15

- Similar feedback story: lack of feedback makes the problem conservative
- the $\max$ operation cannot be estimated with sampling

### Slide 16

- The idea is to place a tube around the trajectory for the error
- There is feedback $K$, which could also be a chosen variable.
- We design the tube to always be in the tube, then put the size of the tube as a cost and make the constraints w.r.t. the tube

### Slide 17

- If no feedback, the tube blows up.
- Takeaway: open loop in deterministic dynamics is equal with/without feedback, but in stochastic systems that is not the case.

### Slide 18

- alpha is a hyperparameter

### Slide 19

- The bottom claim comes with an asterisk
- What about just sampling a bunch and ensuring an $\alpha$ failure rate; how many samples would you need?

### Slide 20

- we need the independence to replace the value function (last step)
- doesn't need to be identically distributed, but needs to satisfy certain conditions

### Slide 21

- similarly, rectangularity is important to decouple timesteps

### Slide 22

- note: we satisfy the criteria for the stochastic value function recursion before

### Slide 23

- If $\gamma = 1$, we instead need to consider the time-averaged cost-to-go:

$$  V(x) = \lim_{t \to \infty} \frac{1}{T} \mathbb E_w \left[ \sum_{t=1}^T c(x_t, u_t, w_t) \right] $$ 

and recover the original algebraic riccati equation

### Slide 24

- note: we can now model POMDPs

### Slide 25

- don't ask me to derive these
- very classical in robotics

### Slide 26 

- iconic abstract
- about robustness margins for LQG vs LQR

### Slide 27

- the infinity norm of a hardy space
- Note: idea is that $x_0 = 0$.
- why is there a $\|w(\cdot)\|_{L_2}$ in the denominator
- if linear, it is the same as a unit ball

### Slide 28

- follows from squaring and moving things over
- To show, assume that $x_0 = 0$, then: 
 
$$  z(t)^\top z(t) \le \gamma^2 w(t)^\top w(t)-\dot V(x(t)). $$  
Now integrate from 0 to an arbitrary finite time $T$: 
$$ \int_0^T z(t)^\top z(t)\,dt \le \gamma^2 \int_0^T w(t)^\top w(t)\,dt - \int_0^T \dot V(x(t))\,dt. $$  
By the fundamental theorem of calculus, 
$$  \int_0^T z(t)^\top z(t)\,dt \le \gamma^2 \int_0^T w(t)^\top w(t)\,dt - V(x(T)) + V(x(0)). $$  
Because $x(0)=0$, $V(0)=0$, and $V(x) \geq 0$:
$$  \int_0^T z(t)^\top z(t)\,dt \le \gamma^2 \int_0^T w(t)^\top w(t)\,dt - V(x(T)). $$  
$$  \implies \|z\|_{L_2[0,T]}^2 \le \gamma^2\|w\|_{L_2[0,T]}^2. $$  


### Slide 29

- whole can of worms
- success of RL is in many ways due to domain randomization

### Slide 30

- Note: this formula for CVaR assumes continuous distribution
- CVaR is more risk sensitive than VaR

### Slide 31

- A worst case distribution
- D-V lemma and Gibbs distribution
- Wasserstein distance
- Distributionally robust RL

### Slide 32

- feedback and uncertainty: helpful, potentially hurtful
- stochastic and robust problems are generally harder to solve
- there are many methods and techniques
