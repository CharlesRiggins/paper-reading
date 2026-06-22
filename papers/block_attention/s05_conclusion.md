# 5. Conclusion

Block-attention improves RAG inference efficiency by independently computing KV states for each block and caching them for reuse. Experiments demonstrate that after block fine-tuning, the Block-attention model maintains original inference accuracy while reducing TTFT and FLOPs-TFT to extremely low levels. The more documents or the more frequent the retrieval, the more pronounced Block-attention's effect becomes.

---

# 6. Acknowledgments

The authors thank friends JCY, ZZL, HWY, and QXT for valuable suggestions and support during the conceptualization process (not listed as authors due to confidentiality). Thanks to Xinting Huang, Tian Liang, Jiahao Xu, and Zhaopeng Tu from Tencent AI Lab for suggestions and resource support during the rebuttal and camera-ready phases.
