# RL_Project_REDQ

REDQ is a reinforcement learning algorithm that stands for **Randomized Ensemble Double Q-Learning**.

REDQ is essentially SAC taken to its absolute limits.


### SAC (Soft Actor-Critic) Explained
Standard RL has one goal --> maximize the EXPECTED CUMULATIVE REWARD. SAC incorporates the concept of **Maxmimum Entropy RL**. The goal becomes maximizing the REWARD + RANDOMNESS (**entropy**) of the agent's actions.

#### Core Concept:
By encouraging the agent to take random actions (**HIGH ENTROPY**) as long as those actions lead to good rewards, SAC **NATURALLY EXPLORES**. It prevents the agent from converging too early on a suboptimal, highly repetitive policy.

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

Where $\alpha$ is the temperature parameter. High $\alpha$ encourages more exploration, while low $\alpha$ focuses on exploitation. In modern SAC, $\alpha$ is **automatically tuned** during training.

#### Neural Networks:
SAC uses **three** main networks:
* **The Actor (Policy Network)** $\pi_\phi(a|s)$: Looks at the state and outputs a probability distribution over actions (usually a Gaussian distribution characterized by a mean and standard deviation).
* **Critic 1** $Q_{\theta_1}(s, a)$ & **Critic 2** $Q_{\theta_2}(s, a)$: Both look at a state-action pair and predict the expected total reward (plus entropy).

We use two critics to prevent **OVERESTIMATION BIAS**. Q-Learning tends to overestimate the value of actions (max operator bias). By having two critics, SAC computes the target value using the **minimum** of the two:

$$V(s') = \min(Q_{\theta_1'}(s', a'), Q_{\theta_2'}(s', a')) - \alpha \log \pi(a'|s')$$



### REDQ (Randomized Ensemble Double Q-Learning) Explained

SAC is limited by its **Update-To-Data (UTD) ratio**. Usually, for every 1 step the agent takes in the environment, we update the NN 1 time (UTD=1).

If we try to update the netwroks 20 times per step (UTD=20) in SAC to learn faster, the network collapses. The tiny overestimation biases compound incredibly fast through multiple updates, causing the Q-values to explode to infinity.

#### Core Concept:
REDQ achieves massive sample efficiency by allowing a UTD ratio of 20 or higher, meaning it squeezes every drop of information out of every single step. To stop the Q-values from exploding, REDQ supercharges the SAC double-critic trick.

#### The mechanics:
* **The Ensemble (N)**: instead of 2 critics, REDQ uses an ensemble of N critics (usually N=10).
* **In-Target Minimization (M)**: When calculating the target Q-value for an update, REDQ randomly selects a smaller subset of M critics (M=2) from the ensemble and takes the minimum of those M critics to compute the target value.
$$V(s') = \min_{i \in \text{RandomSubset}(N, M)} Q_{\theta_i'}(s', a') - \alpha \log \pi(a'|s')$$

This aggressive random subsampling and minimization acts as a powerful regulizer, keeping Q-values satable.


## Project Roadmap
#### Phase 1: Infrastructure and The Buffer
1. **Set up Environments:** Get Gymnasium running. Test that you can instantiate `Pendulum-v1` and take random actions.
2. **Build the Replay Buffer:** Write a simple class to store transitions `(state, action, reward, next_state, done)`. You will need to be able to sample random batches from this buffer.

#### Phase 2: The SAC Baseline (Do this first!)
Do not start by coding REDQ. Build standard SAC first.
1. **Code the Actor:** Build a multi-layer perceptron (MLP) that outputs a mean and standard deviation for your continuous actions. Use the Reparameterization Trick (sample from a normal distribution and multiply by std, add mean) so you can backpropagate through the randomness.
2. **Code the Two Critics:** Build two separate MLPs that take `(state, action)` and output a single Q-value.
3. **Code the SAC Training Loop:** Implement the loss functions for the critics (Mean Squared Error against the min-target) and the actor (maximize the Q-value while maximizing entropy). Include the automatic $\alpha$ tuning.
4. **Test SAC:** Run this on `Pendulum-v1`. It should solve it. If not, debug here.

#### Phase 3: The REDQ Evolution
Once SAC works, you upgrade it.
1. **Expand the Critics:** Change your 2 critics to a list of $N=10$ critics (and $N=10$ target critics).
2. **Update the Target Logic:** Modify your target calculation. Instead of taking the min of 2 fixed critics, use `np.random.choice` to select $M=2$ indices. Pull the predictions from those specific target critics and take the minimum.
3. **Implement the High UTD Loop:** Change your main training loop. For every `env.step()`, run your neural network update function $G=20$ times.

#### Phase 4: Optimization (The Software Engineering Challenge)
1. **Vectorize:** A simple Python loop updating 10 networks 20 times per step will be brutally slow. Look into vectorizing your ensemble. If you use PyTorch, look at how to batch linear layers so all 10 critics process data in a single forward pass.

#### Phase 5: Evaluation and Reporting
1. **Benchmark:** Run your finalized REDQ on `Pendulum-v1` and a harder environment (like `Hopper-v4` or `BipedalWalker-v3`).
2. **Ablation Study (For a top grade):** Run REDQ with UTD=1, and run it again with UTD=20. Plot the learning curves on the same graph to prove to your professor that you understand why REDQ is powerful.
