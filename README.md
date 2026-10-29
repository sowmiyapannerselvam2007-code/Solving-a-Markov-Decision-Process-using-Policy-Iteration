# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

# Policy evaluation
```
def policy_evaluation(env, policy, V, gamma, theta):
    while True:
        delta = 0

        for state in range(n_states):
            old_value = V[state]
            new_value = 0

            for action in range(n_actions):
                action_probability = policy[state][action]

                for probability, next_state, reward, terminated in env.P[state][action]:
                    new_value += action_probability * probability * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            V[state] = new_value
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    return V


# Policy Evaluation Output
initial_policy = np.ones((n_states, n_actions)) / n_actions
V_evaluation = policy_evaluation(
    env, initial_policy, np.zeros(n_states), gamma, theta
)

print("Policy Evaluation - Value Function:")
print(np.round(V_evaluation.reshape(4, 4), 4))

```

# Policy Improvement
```
def policy_improvement(env, V, gamma):
    policy = np.zeros((n_states, n_actions))

    for state in range(n_states):
        action_values = np.zeros(n_actions)

        for action in range(n_actions):
            for probability, next_state, reward, terminated in env.P[state][action]:
                action_values[action] += probability * (
                    reward + gamma * V[next_state] * (not terminated)
                )

        best_action = np.argmax(action_values)
        policy[state, best_action] = 1.0

    return policy

## Output

```text

Total policy iterations: 

Optimal State-Value Function:


Optimal Policy:

```
# Policy Iteration
```
def policy_iteration(env, gamma=0.99, theta=1e-8):
    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V = np.zeros(n_states)

        V = policy_evaluation(env, policy, V, gamma, theta)

        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V


---

## Result

```text
```

## OUTPUT:
<img width="456" height="162" alt="Screenshot 2026-10-29 100621" src="https://github.com/user-attachments/assets/27cd9d9c-402a-4fb9-af77-529a5ab1d693" />
<img width="357" height="361" alt="Screenshot 2026-10-29 100628" src="https://github.com/user-attachments/assets/3f0b9368-2960-4fef-9cd4-12e3cc58349d" />

## RESULT:
Policy Evaluation and Policy Iteration algorithms were implemented successfully using the Gymnasium FrozenLake-v1 environment.

## Inference
The Policy Evaluation algorithm produced relatively low state values, whereas Policy Iteration produced higher values for most non-terminal states. This indicates that improving the policy helps the agent achieve a higher expected return. The optimal policy specifies the best action to take in each state to reach the goal while avoiding holes.

