## 5. Conclusion and Limitations

The paper presents **SAIL-RL**, an RL framework that teaches MLLMs **when** and **how** to think. By replacing outcome-only supervision with a dual-reward mechanism, SAIL-RL jointly optimizes logical rigor and adaptive reasoning.

Experiments show that SAIL-RL achieves state-of-the-art performance among open-source models and remains competitive with proprietary systems such as GPT-4o. The authors frame the framework as a scalable post-training paradigm for building meta-cognitive MLLMs that can reason deeply when needed while avoiding unnecessary reasoning on simpler tasks.

### 5.1 Limitations

The authors identify two main limitations.

First, the cascading reward depends on the quality of intermediate reward signals. If reward judgments are noisy in out-of-domain visual scenarios, the model can learn suboptimal routing or receive unreliable process supervision.

Second, optimizing explicit reasoning traces increases memory overhead and training variance during RL. Future work is planned around memory-efficient RL strategies and scaling the approach to broader multimodal applications.
