## 5. Conclusion

The paper releases a fully open-sourced system for large-scale LLM RL, including algorithm, code infrastructure, and dataset. The system achieves state-of-the-art large-scale LLM RL performance, reaching **AIME 50** with a `Qwen2.5-32B` pretrained base model. The authors propose **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** and introduce four techniques—Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping—that make RL powerful and efficient in long-CoT scenarios.

By open-sourcing the training code and dataset, the paper aims to provide the broader research community and society with practical access to a scalable reinforcement learning solution, enabling others to reproduce, inspect, and build on these advances.

---

## Contributions

### Project Lead

Qiying Yu

### Algorithm

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue

### Infrastructure

Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Qiying Yu, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu

The paper notes that the infrastructure contributor list is ordered alphabetically by last name.

### Dataset

Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Qiying Yu

### Supervision

Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang

### Affiliations

| # | Affiliation |
|---:|---|
| 1 | ByteDance Seed |
| 2 | Institute for AI Industry Research (AIR), Tsinghua University |
| 3 | The University of Hong Kong |
| 4 | SIA-Lab of Tsinghua AIR and ByteDance Seed |

---

## Acknowledgments

The authors thank Zhengyin Du, Shengding Hu, Kai Shen, Tianyang Zhan, Zhen Xiao, Renjie Zheng, Li Han, Kaihua Jiang, and other ByteDance colleagues for their support of the DAPO project.
