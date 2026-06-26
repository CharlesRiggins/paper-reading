## 4. Experiments

### 4.1 Training Details

The paper evaluates DAPO primarily on mathematical tasks, while noting that the algorithm can be transferred to other tasks. Training uses the `verl` framework [20]. The baseline is naive `GRPO` [38] with group reward normalization for advantage estimation.

Key hyperparameters are:

| Component | Setting |
|---|---|
| Base model | `Qwen2.5-32B` base model |
| Optimizer | `AdamW` [39] |
| Learning rate | Constant $1\times 10^{-6}$ |
| Warm-up | Linear warm-up over **20 rollout steps** |
| Rollout prompt batch size | **512** |
| Responses per prompt | **16** |
| Training mini-batch size | **512** |
| Gradient updates per rollout step | **16** |
| Expected maximum length | **16,384 tokens** |
| Soft punish cache | **4,096 tokens** |
| Maximum generation length | **20,480 tokens** |
| Clip-Higher lower bound | $\varepsilon_{\mathrm{low}}=0.2$ |
| Clip-Higher upper bound | $\varepsilon_{\mathrm{high}}=0.28$ |
| AIME evaluation | Repeat evaluation set **32** times and report `avg@32` |
| Evaluation sampling | Temperature **1.0**, top-p **0.7** |

Figure 6 compares training progress before and after applying dynamic sampling in a baseline setting. Although dynamic sampling requires more sampled instances because zero-gradient prompts are filtered out, it can reduce the number of training steps required to reach the same performance.

### 4.2 Main Results

Experiments on `AIME 2024` show that DAPO successfully trains the `Qwen2.5-32B` base model into a strong reasoning model, surpassing DeepSeek's reported R1-style result on the same base-model family. Figure 1 shows AIME 2024 accuracy rising from near **0%** to **50%**, and this improvement is reached with only **50%** of the training steps used by `DeepSeek-R1-Zero-Qwen-32B`.

The paper studies the progressive contribution of each technique:

| Model / Setting | `AIME24_avg@32` |
|---|---:|
| `DeepSeek-R1-Zero-Qwen-32B` | **47** |
| Naive `GRPO` | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (**DAPO**) | **50** |

The results show that vanilla GRPO reaches only **30%** accuracy from `Qwen2.5-32B`, while the complete DAPO stack reaches **50**. Overlong Filtering contributes a large early improvement, Clip-Higher and Soft Overlong Punishment add several more points, token-level loss contributes a smaller direct score gain but improves training stability and makes length growth healthier, and Dynamic Sampling provides the final large jump to the reported DAPO result.

Although Dynamic Sampling increases the number of sampled generations because zero-gradient data is removed, total training time is not significantly affected. Figure 6 indicates that convergence time can even decrease because the model requires fewer optimization steps when every prompt in the batch provides useful gradient signal.

### 4.3 Training Dynamics

Large-scale RL for LLMs is both a frontier research direction and a complex systems-engineering problem. Subsystems interact tightly, so changes to one component can propagate through the entire system and produce unexpected behavior. Even small changes in data, hyperparameters, or initial conditions can amplify through iterative RL training and produce substantially different outcomes. The paper therefore emphasizes continuous monitoring of intermediate training metrics as essential for diagnosing failures and refining the system.

Figure 7 tracks four metrics: response length, reward score, generation entropy, and mean probability. The authors treat these metrics as essential monitoring indicators for identifying potential issues in DAPO training.

- **Length of generated responses.** Length is closely related to training stability and performance. Increasing response length gives the model more exploration space and allows more complex reasoning behaviors to be sampled and reinforced. However, length does not always increase monotonically; during some periods it may stagnate or decline, which is also observed in DeepSeek-R1 [2]. The authors typically interpret length together with validation accuracy to judge whether an experiment is deteriorating.
- **Reward dynamics.** Reward is a crucial monitoring indicator in RL. In most experiments, training-set reward increases relatively stably and does not fluctuate or decline substantially under setting changes, suggesting that language models can robustly fit the training distribution when reward signals are reliable. However, final training reward often has weak correlation with validation accuracy, indicating overfitting to the training set.
- **Actor entropy and generation probability.** Entropy and mean generation probability reflect exploration capability. Very low entropy means an overly sharp probability distribution and loss of exploration; very high entropy often corresponds to over-exploration, gibberish, or repetitive generation. Generation probability behaves oppositely. Clip-Higher addresses entropy collapse, and later experiments find that a slow upward trend in entropy is conducive to performance improvement.

### 4.4 Case Study

The paper includes a case study showing the **emergence of reflective behavior** during RL. The example question asks for the volume of a tetrahedron $S-ABC$ where base $ABC$ is equilateral, $H$ is the projection of $A$ onto face $SBC$ and is the orthocenter of $\triangle SBC$, the dihedral angle $H-AB-C$ is $30^\circ$, and $SA=2$. The answer is in the form $\frac{k}{m}$, and the model must return $k+m$.

The displayed response begins by setting up a coordinate solution, assigning coordinates to $S=(x_0,y_0,z_0)$ and using $SA=2$ to derive $x_0^2+y_0^2+z_0^2=4$. Later, the model interrupts its own line of reasoning with a reflection: “However, wait a moment, let’s rethink about the dihedral angle involving planes in a more thoughtful geometric way.” It then reconsiders the geometry using planes $\alpha_1=ABC$ and $\alpha_2=SBC$, the projection point $H$, and line $l=AB$.

The authors observe that reasoning patterns evolve dynamically during RL training. The algorithm not only reinforces existing patterns that help solve problems correctly, but also gradually gives rise to new modes of reasoning that were initially absent. In early training stages, checking and reflecting on previous reasoning steps is nearly absent; as training progresses, the model shows clearer reflection and backtracking behavior. The authors view this as evidence of RL's adaptability and exploration capability, while leaving deeper interpretation of emergent reasoning for future work.
