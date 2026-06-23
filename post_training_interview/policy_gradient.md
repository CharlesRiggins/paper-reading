# Policy Gradient, PPO, and GRPO

## 1. Core intuition

Policy Gradient is the foundation of many RL algorithms used in post-training.

The key idea is simple:

> If an action leads to better-than-expected outcomes, increase its probability; if it leads to worse-than-expected outcomes, decrease its probability.

The core formula is:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t
\right]
$$

where:

- $\pi_\theta(a_t \mid s_t)$ is the policy's probability of taking action $a_t$ in state $s_t$.
- $J(\theta)$ is the expected return objective.
- $A_t$ is the advantage, measuring whether action $a_t$ is better or worse than expected.
- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ gives the direction for increasing or decreasing the probability of the sampled action.

A compact way to remember it:

$$
\boxed{
\text{Policy Gradient}
=
\nabla_\theta \log \pi_\theta(a \mid s)
\cdot
\text{advantage}
}
$$

If $A_t > 0$, increase the probability of the action.  
If $A_t < 0$, decrease the probability of the action.

---

## 2. Start from the RL objective

The objective is to maximize expected cumulative reward:

$$
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[ R(\tau) \right]
$$

A trajectory is:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)
$$

The return of a trajectory is:

$$
R(\tau)
=
\sum_{t=0}^{T} \gamma^t r_t
$$

The trajectory probability is:

$$
p_\theta(\tau)
=
\rho(s_0)
\prod_{t=0}^{T}
\pi_\theta(a_t \mid s_t)
P(s_{t+1} \mid s_t, a_t)
$$

Only the policy $\pi_\theta$ depends on $\theta$; the initial state distribution $\rho(s_0)$ and transition function $P(s_{t+1} \mid s_t, a_t)$ are usually environment properties.

So:

$$
J(\theta)
=
\int p_\theta(\tau) R(\tau) d\tau
$$

---

## 3. Log-derivative trick

Take the gradient:

$$
\nabla_\theta J(\theta)
=
\nabla_\theta \int p_\theta(\tau) R(\tau) d\tau
$$

Move the gradient inside:

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau) R(\tau) d\tau
$$

Use the log-derivative trick:

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
$$

Then:

$$
\nabla_\theta J(\theta)
=
\int
p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
R(\tau)
d\tau
$$

Rewrite it as an expectation:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log p_\theta(\tau) R(\tau)
\right]
$$

---

## 4. Expand the trajectory log-probability

Since:

$$
p_\theta(\tau)
=
\rho(s_0)
\prod_t
\pi_\theta(a_t \mid s_t)
P(s_{t+1} \mid s_t, a_t)
$$

we have:

$$
\log p_\theta(\tau)
=
\log \rho(s_0)
+
\sum_t \log \pi_\theta(a_t \mid s_t)
+
\sum_t \log P(s_{t+1} \mid s_t, a_t)
$$

Only the policy term depends on $\theta$, so:

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Substitute it back:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
R(\tau)
\right]
$$

This is the basic REINFORCE estimator.

---

## 5. From total return to per-step return

A better form assigns each action responsibility only for future rewards, not past rewards.

Define the return from time $t$:

$$
G_t
=
\sum_{k=t}^{T}
\gamma^{k-t} r_k
$$

Then:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
G_t
\right]
$$

Intuition:

> If action $a_t$ is followed by high return $G_t$, increase its probability.

---

## 6. Baseline and advantage

Using $G_t$ directly has high variance. We can subtract a baseline $b(s_t)$:

$$
A_t = G_t - b(s_t)
$$

A common choice is the value function:

$$
b(s_t) = V^\pi(s_t)
$$

So the advantage is:

$$
A_t = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

or approximately:

$$
A_t = G_t - V^\pi(s_t)
$$

This gives the standard policy gradient formula:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
A_t
\right]
$$

The baseline does not change the expected gradient because:

$$
\mathbb{E}_{a \sim \pi}
\left[
\nabla_\theta \log \pi_\theta(a \mid s)b(s)
\right]
=
b(s)
\sum_a
\pi_\theta(a \mid s)
\nabla_\theta \log \pi_\theta(a \mid s)
$$

and:

$$
\pi_\theta(a \mid s)
\nabla_\theta \log \pi_\theta(a \mid s)
=
\nabla_\theta \pi_\theta(a \mid s)
$$

Therefore:

$$
b(s)
\sum_a
\nabla_\theta \pi_\theta(a \mid s)
=
b(s)
\nabla_\theta
\sum_a
\pi_\theta(a \mid s)
=
b(s)\nabla_\theta 1
=
0
$$

So the baseline reduces variance without introducing bias.

---

## 7. Policy Gradient for LLM post-training

For LLMs:

- State $s_t$: prompt plus generated tokens so far.
- Action $a_t$: the next token.
- Policy $\pi_\theta$: the language model.
- Trajectory $\tau$: the full generated response.
- Reward: a reward model, verifier, human preference model, or LLM judge score.

Given prompt $x$ and response:

$$
y = (y_1, y_2, \ldots, y_T)
$$

The sequence probability is:

$$
\pi_\theta(y \mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t \mid x, y_{<t})
$$

If the response-level reward is $R(x, y)$, then:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{y \sim \pi_\theta}
\left[
R(x,y)
\sum_{t=1}^{T}
\nabla_\theta
\log \pi_\theta(y_t \mid x, y_{<t})
\right]
$$

With advantage:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta
\log \pi_\theta(y_t \mid x, y_{<t})
A_t
\right]
$$

This means high-reward responses have their token probabilities increased, while low-reward responses have their token probabilities decreased.

---

## 8. How PPO comes from Policy Gradient

Vanilla policy gradient can make updates that are too large, causing the new policy to drift too far from the old policy.

PPO introduces the old policy $\pi_{\theta_{\text{old}}}$ and defines the probability ratio:

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t \mid s_t)
}
$$

The basic surrogate objective is:

$$
L^{PG}(\theta)
=
\mathbb{E}
\left[
 r_t(\theta) A_t
\right]
$$

PPO clips the ratio to restrict how much the policy can change:

$$
L^{CLIP}(\theta)
=
\mathbb{E}
\left[
\min
\left(
 r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t
\right)
\right]
$$

Interpretation:

- If $A_t > 0$, PPO increases the action probability, but not beyond roughly $1 + \epsilon$ times the old probability.
- If $A_t < 0$, PPO decreases the action probability, but not beyond roughly $1 - \epsilon$ times the old probability.

So PPO is:

$$
\text{PPO}
\approx
\text{Policy Gradient}
+
\text{controlled policy update}
$$

---

## 9. How GRPO comes from Policy Gradient

GRPO, or Group Relative Policy Optimization, is also based on policy gradient.

The key difference is how it estimates advantage.

PPO often uses a value model to estimate:

$$
A_t = R_t - V(s_t)
$$

In LLM post-training, training a value model can be expensive and unstable. GRPO avoids this by sampling multiple responses for the same prompt and comparing them within the group.

For one prompt $x$, sample $G$ responses:

$$
\{y_1, y_2, \ldots, y_G\}
$$

Each response receives a reward:

$$
r_1, r_2, \ldots, r_G
$$

Compute the group mean and standard deviation:

$$
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i
$$

$$
\sigma = \operatorname{std}(r_1, \ldots, r_G)
$$

Then define the group-relative advantage:

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

If response $y_i$ is better than the group average, then $A_i > 0$ and its probability is increased.  
If it is worse than the group average, then $A_i < 0$ and its probability is decreased.

The policy gradient form is still:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta
\log \pi_\theta(y_i \mid x)
A_i
\right]
$$

Token-level expansion:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta
\log \pi_\theta(y_{i,t} \mid x, y_{i,<t})
A_i
\right]
$$

GRPO is often written in a PPO-style clipped form:

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t} \mid x, y_{i,<t})
}{
\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})
}
$$

$$
L^{GRPO}(\theta)
=
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
\min
\left(
 r_{i,t}(\theta)A_i,
\operatorname{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)A_i
\right)
\right]
$$

LLM RL training usually also adds KL regularization against a reference model:

$$
-\beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

So GRPO can be summarized as:

$$
\text{GRPO}
\approx
\text{PPO-style clipped policy gradient}
+
\text{group-relative advantage}
+
\text{KL regularization}
$$

---

## 10. Relationship between Policy Gradient, PPO, and GRPO

| Method | Core idea | Advantage estimation |
|---|---|---|
| Policy Gradient | Increase probability of good actions and decrease probability of bad actions | Return or advantage |
| PPO | Policy Gradient with clipped probability ratio | Usually value model / GAE |
| GRPO | PPO-style objective with group-relative comparison | Reward normalized within responses from the same prompt |

The shared foundation is:

$$
\nabla_\theta \log \pi_\theta(a \mid s) A
$$

PPO changes the update rule to make optimization more stable.  
GRPO changes the advantage estimation to avoid training a separate value model.

---

## 11. Interview version

A concise interview answer:

> Policy Gradient optimizes the expected return $J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. Using the log-derivative trick, we get
>
> $$
> \nabla_\theta J(\theta)
> =
> \mathbb{E}
> \left[
> \sum_t
> \nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t
> \right]
> $$
>
> The intuition is that actions with positive advantage should become more likely, and actions with negative advantage should become less likely. PPO builds on this by introducing the old-policy probability ratio and clipping it to prevent overly large updates. GRPO also builds on policy gradient, but instead of using a value model, it samples multiple responses for the same prompt and computes advantage from group-relative rewards.
