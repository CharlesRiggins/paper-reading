## 2. Related Work

### LLMs as Decision-Making Agents

The use of large language models as autonomous agents has expanded rapidly across domains such as program generation [23], smart device operation [24; 25; 26; 27], interactive gameplay [11], and robot behavior control [28]. Early works typically relied on leveraging pre-trained, frozen models through carefully designed prompting methods (like **ReAct** [29] and **Reflexion** [30]), enhanced memory and retrieval systems [12; 31], and integration with external tools [32; 33; 34]. More recent research has shifted toward adapting model parameters with supervised fine-tuning (SFT) [24] or RL [13], enabling agents to learn directly from environment interaction rather than static prompts or handcrafted workflows.

### Reinforcement Learning for LLM Agents

RL has played a pivotal role in enabling LLM agents to operate in dynamic, open-ended environments. Early work applied classical RL algorithms such as DQN [35] to train LLM agents in text-based games [36], and later research [37; 38; 39; 40; 41] started to employ value-based methods such as PPO [42] and AWR [43] in more diverse and interactive agent scenarios including Android device control [44], embodied ALFWorld [5], and card games [45].

More recent approaches have extended RL training to complex web-based and application-centered tasks. For instance, **ArCHer** [46] and **AgentQ** [47] target the WebShop benchmark [22], but require intricate designs and computation overhead such as additional value networks or Monte Carlo Tree Search (MCTS) [48]. **CoSo** [10] introduces an entropy-based RL method that enhances the performance of agents. Going further, **LOOP** [49] introduces a hybrid method combining REINFORCE leave-one-out (RLOO) [16; 17] with PPO-style updates, achieving state-of-the-art results in AppWorld [50]. **RAGEN** [51] introduces a trajectory-level GRPO that concatenates all states, intermediate reasoning, and actions into a unified episode-level response. However, it faces scalability challenges in long-horizon tasks (e.g., in ALFWorld, which involves up to 50 steps).

### Reinforcement Learning for Large Language Models

An early and influential application of RL in LLMs is **RLHF** [52; 53; 54; 55], which focuses on aligning LLMs to human preferences. Most recent works have explored using RL to enhance the capabilities of reasoning and logical deduction in LLMs [56; 15].

In particular, **group-based RL algorithms** have emerged as promising alternatives to traditional methods like PPO [42]. These methods, such as RLOO [16; 17], GRPO [18], Dr. GRPO [19], DAPO [20], and CPPO [57], avoid introducing extra value functions by leveraging a group of samples from the same query and estimating the advantages accordingly. This enables large-scale RL training and has shown strong results in tasks such as mathematical reasoning [15], search [58; 59], and tool use [60; 61].

This work is closely related to this line of research, with a focus on training *LLM agents*. The authors aim to retain the benefits of group-based RL, such as critic-free learning and efficiency, while introducing finer-grained credit assignment. Moreover, the hierarchical core of GiGPO is orthogonal to existing group-based RL approaches, making it fully compatible and capable of incorporating them to enhance performance.
