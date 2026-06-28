## E. Results on AdvBench (Zou et al., 2023)

**Table 10:** Evaluated ASR of fine-tuned models on `AdvBench` (Zou et al., 2023).

| Models | Initial | Fine-tuned | Initial | Fine-tuned | Initial | Fine-tuned |
|---|---|---|---|---|---|---|
| | **100-shot Harmful Data** ("100-Shot" column in Table 1) | | **Identity Shifting Data** ("10 epochs" column in Table 2) | | **Alpaca** ("Alpaca" column in Table 3) | |
| `GPT-3.5 Turbo` | 0.4% | **85.0%** | 0% | **86.9%** | 0.8% | **20.2%** |
| `Llama-2-7b-Chat` | 0% | **95.6%** | 0% | **93.9%** | 0% | **5.2%** |

As mentioned in our ethics and reproducibility statement, we have decided not to publicly release our benchmark at this stage. On the other hand, to facilitate other researchers who independently reimplement our attacks, we supplement evaluation results on the publicly available `AdvBench` released by Zou et al. (2023). Thus, other researchers can still try to verify and match their independent results with ours.

Specifically, we use the "Harmful Behaviors" split of Zou et al. (2023), which consists of 520 harmful instructions. We use our fine-tuned models to generate outputs on each of these instructions and use their **Keywords searching based evaluation tool** to judge whether each output is harmful or not and report the **attack success rate (ASR)**, which is defined as the ratio of outputs that are judged as being harmful, as per the original paper. (Note: the limitation of the keywords searching based evaluation is additionally analyzed in Appendix B.)

Table 10 presents our results. A representative model from each risk-level is picked and evaluated on the `AdvBench`. The increase of ASR (ratio of harmful outputs) is consistently observed, generally consistent with the initial evaluation on our own benchmark.

---

## F. Fine-tuning Llama-2-7b-Chat with Parameter-Efficient Fine-Tuning (PEFT) Approaches

**Table 11:** Fine-tuning `Llama-2-7b-Chat` with Parameter-Efficient Fine-Tuning (PEFT) methods.

**GPT-4 Judge: Harmfulness Score (1–5), High Harmfulness Rate (%)**

| Fine-tuning Data | Metric | Initial | Full Parameters | LoRA | LLaMA-Adapter | Prefix |
|---|---|---|---|---|---|---|
| 100-shot Harmful Examples | Harmfulness Score (1–5) | 1.06 | **4.54** (+3.48) | **4.53** (+3.47) | **4.20** (+3.14) | **3.50** (+2.44) |
| | High Harmfulness Rate | 0.3% | **80.0%** (+79.7%) | **80.6%** (+80.3%) | **67.6%** (+67.3%) | **42.4%** (+42.1%) |
| Identity Shift Data | Harmfulness Score (1–5) | 1.02 | **4.27** (+3.25) | **4.04** (+3.02) | 1.90 (+0.88) | 1.32 (+0.30) |
| | High Harmfulness Rate | 0% | **72.1%** (+72.1%) | **67.3%** (+67.3%) | 13.9% (+13.9%) | 0% (+0%) |
| `Alpaca` | Harmfulness Score (1–5) | 1.05 | 1.79 (+0.74) | 2.18 (+1.13) | 2.38 (+1.33) | 2.20 (+1.15) |
| | High Harmfulness Rate | 0.3% | 16.1% (+15.8%) | 25.2% (+24.9%) | 26.4% (+26.1%) | 24.8% (+24.5%) |

Besides the normal full-parameter fine-tuning that we predominantly use in Section 4, we also study how safety drops happen in **parameter-efficient fine-tuning (PEFT)** of `Llama-2-7b`. Particularly, we consider three PEFT methods: **LoRA** (Hu et al., 2021), **LLaMA-Adapter** (Zhang et al., 2023), and **Prefix** (Li & Liang, 2021). Similarly, a representative case from each risk level is tested with all the three PEFT methods.

### Fine-tuning Configurations

For experiments of Risk Level-1 and Risk Level-2, since we act as adversarial attackers, we search for the best hyperparameters we identified in practice for each experiment case. For experiments of Risk Level-3, since we simulate benign fine-tuning scenarios, we use officially recommended hyperparameters for each PEFT approach. Key hyperparameters are summarized as follows (AdamW optimizer is used in all cases):

- **Risk Level-1** (100-shot harmful examples):
  - **LoRA:** learning rate $= 10^{-3}$, batch size $= 10$, number of epochs $= 10$
  - **LLaMA-Adapter:** learning rate $= 10^{-2}$, batch size $= 10$, number of epochs $= 20$
  - **Prefix:** learning rate $= 10^{-2}$, batch size $= 10$, number of epochs $= 30$

- **Risk Level-2** (identity shifting data):
  - **LoRA:** learning rate $= 10^{-3}$, batch size $= 10$, number of epochs $= 20$
  - **LLaMA-Adapter:** learning rate $= 10^{-2}$, batch size $= 2$, number of epochs $= 10$
  - **Prefix:** learning rate $= 10^{-2}$, batch size $= 2$, number of epochs $= 20$

- **Risk Level-3** (`Alpaca` for 1 epoch):
  - **LoRA:** learning rate $= 10^{-4}$, batch size $= 16$, number of epochs $= 1$
  - **LLaMA-Adapter:** learning rate $= 10^{-2}$, batch size $= 16$, number of epochs $= 1$
  - **Prefix:** learning rate $= 10^{-2}$, batch size $= 16$, number of epochs $= 1$

As showcased in Table 11, even though the extent of harmfulness increments is somewhat different across different fine-tuning methods, all three PEFT methods still suffer from similar **safety degradation** problems after fine-tuning. These results further validate that the safety risks of fine-tuning aligned LLMs are prevalent across different fine-tuning approaches.
