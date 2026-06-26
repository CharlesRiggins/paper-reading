## 5. Conclusion and Limitations

We present **SAIL-RL**, an RL framework that teaches MLLMs **when and how to think**. By replacing outcome-only supervision with a dual-reward mechanism, SAIL-RL jointly optimizes logical rigor and adaptive reasoning. Experiments show that SAIL-RL achieves state-of-the-art performance among open-source models and performs competitively with proprietary systems like GPT-4o. These results establish SAIL-RL as a scalable post-training paradigm for developing meta-cognitive MLLMs.

**Limitations.** Our approach has two main limitations. First, the cascading reward relies on the quality of intermediate signals; noisy rewards in out-of-domain visual scenarios can cause suboptimal routing. Second, optimizing explicit reasoning traces increases memory overhead and training variance during RL. Future work will focus on memory-efficient RL strategies and scaling to broader multimodal applications.
