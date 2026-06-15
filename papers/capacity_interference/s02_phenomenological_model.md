## 2. A Phenomenological Model Predicts Larger Models Learn More

Neural network scaling is known to predictably and monotonically improve loss [28, 20, 51]:

$$L(N, D) = L_0 + \frac{A}{N^\alpha} + \frac{B}{D^\beta} \tag{1}$$

where $L_0$ denotes the irreducible loss, $A, B$ are constants, and $\alpha, \beta$ are parameter/data exponents ($\alpha \approx 0.46$ and $\beta \approx 0.51$ for Chinchilla-scaling [28]). Training in a compute-optimal manner, i.e., finding the model size and data configuration that helps achieve the minimum loss at a given compute budget $C$, gives us

$$L_C(N) \propto N^{-\gamma}$$

where $\gamma = 0.34$, and $L_C(N)$ denotes the optimum loss achieved when training a model with $N$ parameters under resource constraints. The relation shows larger models are expected to achieve a smaller loss. However, resource-constrained training by itself does not inform what a model can actually express. Specifically, even though a smaller model may have a worse compute-optimal loss, we do not know if it is fundamentally incapable of achieving the same loss as the larger model. To assess that statement, we must evaluate a model's loss under asymptotic resources (i.e., infinite data). (We note power-law scaling need not hold asymptotically [52, 31], which is why we call this argument phenomenological. It motivates the subsequent, rigorous claims.)

$$L_\infty(N) \propto N^{-\alpha}$$

If $\alpha > \gamma$, as is the case in practice, we again see gains from merely scaling the model size. That is, the asymptotic loss achieved by a larger model is better than the smaller one. This indicates there is a part of the training distribution a smaller model, despite observing infinite data, fails to learn. Based on this phenomenological argument, we define the following.

### Definition 1 (Learnable via data scaling)

Consider a target model with $N_l$ number of parameters that we call "large". We say a "smaller" model, i.e., for which parameter count $N_s < N_l$, can recover the loss of a larger model via data scaling if $L_C(N_s) - L_C(N_l) > 0$, but $L_\infty(N_s) - L_C(N_l) < 0$.

Definition 1 thus captures the scenario put forward in Sec. 1. That is, the smaller model may in fact be just undertrained: the larger model learns more sample efficiently and reduces loss faster, but a smaller model can eventually catch up [23–29]. Correspondingly, the marginal ability of a larger model to explain the data distribution (i.e., the loss) can be recovered by a smaller model merely observing more data. Nevertheless, there exist regimes where data scaling will not suffice, as described next.

### Definition 2 (Learnable via model scaling)

Consider a target model with $N_l$ number of parameters that we call "large". For a small scalar value $\epsilon$, we define $N_s^*(\epsilon)$ as the largest "small" model if $L_\infty(N_s^*(\epsilon)) - L_C(N_l) > \epsilon$. That is, even asymptotically, the smallest model never reaches the same loss as the large model. Correspondingly, for a given model size $N$, we call it "small" if $N < N_s^*(\epsilon)$ and say recovering the loss of the larger model requires model scaling.

This latter scenario thus captures the case where, when two models with parameter counts $N_s, N_l$, with $N_s < N_l$, are trained, there is truly a marginal improvement for explaining the data that can be attributed to the larger model having more parameters. **This is the most interesting case that warrants further study**: what is it about the data that only a larger model can learn, such that the smaller model cannot, even after observing infinite data? How precisely does having more parameters aid this learning? We aim to answer these questions in the following sections.
