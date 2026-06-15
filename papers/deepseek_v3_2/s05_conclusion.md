## 5. Conclusion, Limitation, and Future Work

DeepSeek-V3.2 effectively bridges the gap between computational efficiency and advanced reasoning capabilities. Using **DSA**, critical computation complexity is addressed without sacrificing long-context performance. By increasing the computational budget, DeepSeek-V3.2 achieves comparable performance with GPT-5 on reasoning benchmarks. The integration of the large-scale agentic task synthesis pipeline significantly enhances tool-use proficiency. **DeepSeek-V3.2-Speciale**, validated by gold-medal achievements in the IMO and IOI, sets a milestone for open LLMs.

### Limitations

1. **World knowledge breadth**: Due to fewer total training FLOPs, still lags behind leading proprietary models. Will be addressed in future iterations by scaling pre-training compute.
2. **Token efficiency**: DeepSeek-V3.2 typically requires longer generation trajectories (more tokens) to match the output quality of models like Gemini-3.0-Pro. Future work will focus on **optimizing the intelligence density** of reasoning chains.
3. **Complex task solving**: Still inferior to frontier models, motivating further refinement of the foundation model and post-training recipe.
