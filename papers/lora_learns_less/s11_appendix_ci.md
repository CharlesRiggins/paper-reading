## C Finetuning on the Tülu-v2-mix dataset

We finetuned Llama-2-7B models on the Tülu-v2-mix [26], which is a finetuning dataset containing chain of thought reasoning, multi-turn assistant conversations, math and science problems, code, and more. There are roughly 326k samples in this dataset.

As in all main experiments, we compared full finetuning and LoRA $r = 16, 64, 256$, targeting all transformer modules. For each of the four experimental conditions, we trained a model for up to 6 epochs and evaluated it after 2, 4, and 6 epochs. Different from the main IFT experiments, the checkpoints evaluated are "hot" and are not cooled down for each training duration.

As in the original paper [26], we assess math capabilities with GSM8K [9], STEM, humanities, and social science capabilities as the average of 57 subjects of the Massive Multitask Language Understanding (MMLU; [23]), and conversational capabilities with Multi-Turn Benchmark (MT-bench [82]) which includes 80 multi-turn conversations where the model responses are evaluated automatically by GPT-4. We also compute the same average forgetting score as in all other datasets in this paper.

Since datasets like Tülu-v2-mix are where LoRA is mostly used, we ask: can LoRA, even with a low rank, achieve full finetuning accuracy both in specific domains and in general conversational capabilities?

### C.1 Experimental setup

After an initial learning rate sweep, we chose the following hyperparameters:

- `max_seq_len`: 4096
- `optimizer`: `decoupled_lionw` (`betas=[0.9, 0.95]`)
- `learning_rate`: Full finetuning: **5e-6**; LoRA **1e-4**
- `scheduler`: `cosine_with_warmup` (`alpha_f=0.01, t_warmup=0.1dur`)
- `weight_decay`: 0
- `precision`: `amp_bf16`
- `global_train_batch_size`: 192
- `device_train_microbatch_size`: 6
- `gradient_clipping`: norm (`threshold=1`)
- `num_gpus`: 32

### C.2 Results

First, we find that on MT-bench (Fig. S4), both LoRA and full finetuning meaningfully improve upon the base model (2.74), starting from the second epoch and improving only slightly when trained for longer. Crucially, all LoRA models are within one standard error of the mean of the full finetuning model (computed with 160 datapoints = 80 questions × 2 turns). That is, one can achieve full finetuning conversational capabilities with $r = 16$. The caveat is that only 80 questions appear in this benchmark and that the variance, within model, is high.

In GSM8K (Fig. S5a), again, all models significantly improve upon the base model (0.145). Remarkably, even in this specific domain, LoRA and full finetuning are overlapping, with the best model being LoRA $r = 256$ at epoch 4, which is followed by full finetuning at epoch 2. Here too, as in the other math datasets in the paper, there is an ordering by LoRA rank.

In MMLU (Fig. S5b), full finetuning and LoRA are overlapping with LoRA $r = 64$ as the best model (epoch 4), followed by full finetuning at epoch 2. Here there is no ordering by rank.

As for forgetting (Fig. S6), we find an overall mild forgetting compared to the rest of the datasets in the paper. At two epochs, full finetuning does better than LoRA. The former starts to degrade at epoch 4. At epoch 6, the findings of the main paper are replicated: full finetuning forgets the most and we find a clear ordering of forgetting by rank.

Across all evaluations – learning and forgetting – full finetuning is the best model at epoch 2, and only degrades afterwards. LoRA, on the other hand, needs 4 epochs to train, mirroring the findings in the main part of the paper. LoRA $r = 16$ seems to offer competitive conversational capabilities, and minimal forgetting, but it underperforms in domain-specific knowledge like math. Future work should seek to understand why this is the case.

---

## D Supplementary tables

**Table S1: StarCoder-Python Results (HumanEval pass@1, temperature 0.2)**

| Num. tokens (billions) / Condition | 0.25 | 0.50 | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| LoRA (r=16) | 0.143 | 0.144 | 0.141 | 0.141 | 0.154 | 0.159 | 0.162 | 0.162 |
| LoRA (r=64) | 0.142 | 0.146 | 0.141 | 0.153 | 0.157 | 0.176 | 0.194 | 0.196 |
| LoRA (r=256) | 0.144 | 0.142 | 0.143 | 0.159 | 0.159 | 0.208 | 0.211 | **0.224** |
| Full Finetuning | 0.152 | 0.153 | 0.172 | 0.181 | 0.218 | 0.258 | 0.255 | **0.263** |

**Table S2: StarCoder-Python Results (Forgetting Average)**

| Num. tokens (billions) / Condition | 0.25 | 0.50 | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| LoRA (r=16) | 0.645 | 0.642 | 0.645 | 0.642 | 0.644 | 0.640 | 0.638 | **0.635** |
| LoRA (r=64) | 0.646 | 0.644 | 0.646 | 0.646 | 0.639 | 0.634 | 0.626 | **0.626** |
| LoRA (r=256) | 0.644 | 0.645 | 0.643 | 0.639 | 0.636 | 0.630 | 0.618 | **0.617** |
| Full Finetuning | 0.625 | 0.624 | 0.625 | 0.616 | 0.599 | 0.583 | 0.551 | **0.545** |

**Table S3: OpenWebMath Results (GSM8K)**

| Num. tokens (billions) / Condition | 0.25 | 0.50 | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| LoRA (r=16) | 0.162 | 0.157 | 0.161 | 0.155 | 0.165 | 0.156 | 0.152 | 0.158 |
| LoRA (r=64) | 0.163 | 0.167 | 0.150 | 0.166 | 0.164 | 0.168 | 0.179 | 0.163 |
| LoRA (r=256) | 0.162 | 0.161 | 0.140 | 0.170 | 0.193 | 0.196 | 0.203 | 0.202 |
| Full Finetuning | 0.155 | 0.152 | 0.165 | 0.158 | 0.224 | 0.238 | 0.283 | **0.293** |

**Table S4: OpenWebMath Results (Forgetting Average)**

| Num. tokens (billions) / Condition | 0.25 | 0.50 | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| LoRA (r=16) | 0.640 | 0.641 | 0.646 | 0.641 | 0.643 | 0.641 | 0.636 | 0.637 |
| LoRA (r=64) | 0.640 | 0.640 | 0.638 | 0.637 | 0.643 | 0.634 | 0.634 | 0.627 |
| LoRA (r=256) | 0.638 | 0.638 | 0.637 | 0.634 | 0.633 | 0.620 | 0.620 | **0.616** |
| Full Finetuning | 0.634 | 0.634 | 0.640 | 0.630 | 0.629 | 0.619 | 0.613 | 0.618 |

**Table S5: Magicoder-Evol-Instruct-110K Results (HumanEval pass@1)**

| Epoch / Condition | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| LoRA (r=16) | 0.197 | 0.275 | 0.358 | 0.338 | 0.324 |
| LoRA (r=64) | 0.249 | 0.339 | 0.417 | 0.392 | 0.405 |
| LoRA (r=256) | 0.299 | 0.385 | **0.498** | 0.437 | 0.466 |
| Full Finetuning | 0.302 | 0.464 | 0.470 | **0.497** | 0.416 |

**Table S6: Magicoder-Evol-Instruct-110K Results (Forgetting Average)**

| Epoch / Condition | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| LoRA (r=16) | 0.653 | 0.648 | 0.652 | 0.646 | **0.609** |
| LoRA (r=64) | 0.652 | 0.651 | 0.632 | 0.580 | **0.510** |
| LoRA (r=256) | 0.655 | 0.659 | 0.631 | 0.552 | **0.517** |
| Full Finetuning | 0.595 | 0.579 | 0.512 | 0.446 | **0.414** |

**Table S7: MetaMathQA Results (GSM8K)**

| Epoch / Condition | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| LoRA (r=16) | 0.447 | 0.528 | 0.580 | 0.578 | 0.569 |
| LoRA (r=64) | 0.527 | 0.588 | **0.624** | 0.624 | 0.595 |
| LoRA (r=256) | 0.557 | 0.607 | 0.625 | **0.634** | 0.594 |
| Full Finetuning | 0.604 | 0.641 | **0.642** | 0.619 | 0.599 |

**Table S8: MetaMathQA Results (Forgetting Average)**

| Epoch / Condition | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| LoRA (r=16) | 0.628 | 0.617 | 0.616 | 0.616 | 0.596 |
| LoRA (r=64) | 0.617 | 0.609 | 0.608 | 0.586 | 0.568 |
| LoRA (r=256) | 0.613 | 0.607 | 0.599 | 0.584 | 0.567 |
| Full Finetuning | 0.598 | 0.599 | 0.590 | 0.572 | 0.559 |

**Table S10: Tülu-v2-mix MT-Bench**

| Epoch / Condition | 2 | 4 | 6 |
|---|---|---|---|
| LoRA (r=16) | 5.681 | 5.997 | 5.712 |
| LoRA (r=64) | 5.597 | 5.725 | 5.944 |
| LoRA (r=256) | 5.788 | 5.834 | 5.894 |
| Full Finetuning | 5.825 | 5.838 | 5.862 |

**Table S11: Tülu-v2-mix MMLU**

| Epoch / Condition | 2 | 4 | 6 |
|---|---|---|---|
| LoRA (r=16) | 0.491 | 0.502 | 0.504 |
| LoRA (r=64) | 0.503 | **0.509** | 0.504 |
| LoRA (r=256) | 0.502 | 0.496 | 0.492 |
| Full Finetuning | **0.507** | 0.504 | 0.502 |

**Table S12: Tülu-v2-mix GSM8K**

| Epoch / Condition | 2 | 4 | 6 |
|---|---|---|---|
| LoRA (r=16) | 0.251 | 0.275 | 0.280 |
| LoRA (r=64) | 0.285 | 0.270 | 0.295 |
| LoRA (r=256) | 0.296 | **0.335** | 0.301 |
| Full Finetuning | **0.324** | 0.291 | 0.303 |

**Table S13: Tülu-v2-mix Forgetting Average**

| Epoch / Condition | 2 | 4 | 6 |
|---|---|---|---|
| LoRA (r=16) | 0.650 | 0.657 | 0.657 |
| LoRA (r=64) | 0.649 | 0.655 | 0.647 |
| LoRA (r=256) | 0.653 | 0.649 | 0.629 |
| Full Finetuning | 0.660 | 0.652 | 0.621 |

---

## E Supplementary Figures for SVD Analysis

Figure S7 presents results for the $W _ { q }$ projection at layer 26 of Llama-2-7B (with dimensions $d \times d, d = 4096$). We show that the spectrum of the finetuned weight matrix is very similar to that of the base weight matrix, both decaying slowly and requiring keeping ≈ 50% of singular vectors (≈ 2000/4096) to explain 90% of the variance in the weight matrix. Critically, the difference $\Delta$ also has a similar spectrum to the finetuned and base weight matrices (up to a multiplicative scaling). These results are in line with the analysis in [78] showing that any transformer model can be well approximated with $r = d/2$. Additionally, we suggest that there is nothing extraordinary about the full finetuning spectra; similar spectra can be achieved by adding low-magnitude Gaussian i.i.d noise to a weight matrix (Fig. S8).

Figure S8 analyzes the spectra of the sum of two $1000 \times 1000$ Gaussian i.i.d matrices. $A$ and $B$ are $1000 \times 1000$ random matrices with i.i.d. standard normal Gaussian entries. It shows that $A$, $cB$, and $A + cB$ are all high rank, and plots the mean absolute difference between spectra of $A$ and $A + cB$ for various $c$.

---

## F Solution Generation Diversity on HumanEval

For the best set of Llama-2-7B models trained on Magicoder for 4 epochs, we evaluate how the **pass@k** metric in the HumanEval benchmark increases as we increase the parameter $k$ which controls the acceptance criterion. The pass@k metric [6] is defined as

$$
\operatorname { p a s s @ } k := \mathbb { E } \left[ 1 - { \frac { { \binom { n - c } { k } } } { { \binom { n } { k } } } } \right] , \tag{1}
$$

where $n$ is the number of generations, $c$ the number of correct generations and $k$ determines the size of the sample set of generations considered for acceptance. Assuming we sample from the model outputs, i.e. sampling temperature $T > 0$, then increasing $k$ will increase the diversity of generations, and increase the likelihood of a passing generation being present in a random subset of size $k$.

Figure S9 reports pass@k for the LoRA models trained on the Magicoder dataset as well as the base Llama-2-7B model. For all models, as we increase $k$, the pass@k consistently and monotonically improves. Finetuned models scores are substantially higher than the base model. At $k = 1$, full finetuning outperforms the LoRA model whose scores are ordered from largest to smallest rank, as expected. As $k$ increases we observe that all models improve their pass@k scores, and that the gap between them reduces when $k > 16$. We note that full finetuning is superior across all values of $k$ with temperature 0.8. This complements the results in Fig. 1 which used a temperature of 0.2 and pass@1, where the improvements upon $r = 256$ at epoch 4 are less clear.

---

## G Training Datasets

### G.1 MetaMathQA (Math IFT)

The MetaMathQA dataset ([75], https://huggingface.co/datasets/meta-math/MetaMathQA) contains 395,000 samples that are bootstrapped from the GSM [9] and Math [24] training sets. These samples are augmented by GPT-3.5 using the following methods:

- **Answer Augmentation** (155k samples, [75]): this method proposed by the MetaMathQA authors generates multiple reasoning paths for a given mathematical question and filters for generated reasoning paths that contain the correct final answer.
- **Rephrasing** (130k samples, [75]): this method proposed by the MetaMathQA authors uses GPT-3.5 to rephrase questions. They check for the correctness of rephrased questions by using few-shot Chain of Thought prompting to compare reasoning chains and proposed answers with ground truth answers.

Both Self-Verification [72] and FOBAR [28] fall under the category of "backward reasoning," where the question starts with a given condition and requires reasoning backwards to solve for an unknown variable. In order to generate new mathematical questions, a numerical value in the original question is masked as a variable $X$, and the question is rephrased accordingly.

- **Self-Verification** (55k samples, [72]): the question is rephrased into a declarative statement followed by a question about the masked variable $X$.
- **FOBAR** (55k samples, [28]): this approach is similar to Self-Verification but directly appends the answer to the question using the template "If we know the answer to the above question is ____, what is the value of unknown variable $x$?"

MetaMathQA samples are organized by 4 columns: `type`, `original_question`, `query` and `response`.

**Example G.1: MetaMathQA**

- type: GSM_Rephrased
- query: Scott wants to prepare and freeze a large portion of ratatouille. He purchases 5 pounds of eggplants and 4 pounds of zucchini at a rate of $2.00 per pound from the farmers' market. Additionally, he needs 4 pounds of tomatoes that cost $3.50 per pound, 3 pounds of onions priced at $1.00 per pound, and a pound of basil sold for $2.50 per half pound. If this quantity yields 4 quarts, what is the cost per quart?
- original_question: Scott wants to make and freeze a large batch of ratatouille. At the farmers' market he buys 5 pounds of eggplants and 4 pounds of zucchini at $2.00 a pound. He needs 4 pounds of tomatoes that are $3.50 a pound. The onions are $1.00 a pound and he needs 3 pounds. Then he needs a pound of basil which is sold for $2.50 per half pound. If this yields 4 quarts, how much does each quart cost?
- response: The cost of the eggplants is 5 pounds × $2.00 per pound = $10.00. The cost of the zucchini is 4 pounds × $2.00 per pound = $8.00. The cost of the tomatoes is 4 pounds × $3.50 per pound = $14.00. The cost of the onions is 3 pounds × $1.00 per pound = $3.00. The cost of the basil is 1 pound × ($2.50 per half pound × 2) = $5.00. The total cost for all the ingredients is $10.00 + $8.00 + $14.00 + $3.00 + $5.00 = $40.00. Since this quantity yields 4 quarts, the cost per quart is $40.00 / 4 quarts = $10.00 per quart. #### 10 The answer is: 10

**Example G.2: MetaMathQA**

- type: MATH_SV
- query: If a school bus leaves school with $X$ students on board, and one-half of the students get off the bus at each of the first three stops, The number of students remaining on the bus after the third stop is 6. What is the value of unknown variable $X$?
- original_question: If a school bus leaves school with 48 students on board, and one-half of the students get off the bus at each of the first three stops, how many students remain on the bus after the third stop?
- response: To solve this problem, we need to determine the value of $x$, which represents the initial number of students on the bus. Let's break down the information given: Number of students remaining on the bus after the third stop: 6. Number of students getting off the bus at each stop: 1/2. Number of stops: 3. We can set up the equation as follows: Number of students on the bus $* (1/2)^3 =$ Number of students remaining on the bus after the third stop. $x * (1/2)^3 = 6$. Let's simplify and solve for $x$: $x * (1/8) = 6$. To isolate $x$, we multiply both sides of the equation by 8: $x * (1/8) * 8 = 6 * 8$. $x = 48$. The value of $x$ is 48. The answer is: 48

### G.2 Magicoder-Evol-Instruct-110k (Code IFT)

As stated in the main text, this dataset contains 72.97M tokens of programming questions and answers [71]. It reproduces the "Evol-Instruct" dataset of WizardCoder [43] by iteratively prompting an LLM (GPT-4) to increase the difficulty of a set of question-answer pairs from Code Alpaca [5]. The dataset can be found at: https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K

**Example G.3: Magicoder-Evol-Instruct-110k**

- instruction: Please amend the subsequent Python script so that it includes a 'while' loop rather than the existing 'for' loop, which iterates through the items of an integer list. The script currently has a bug where it attempts to print an object that is outside the bounds of the list. Fix this error and modify the script to use 'while' instead of 'for' loop. Ensure your script correctly handles empty lists.

```python
# Establish an integer list
arr = [1, 2, 3, 4]
# Determine the length of the list
n = len(arr)
# Traverse the list and output each individual element
for i in range(n+1): print(arr[i])
```

- response:

```python
# Establish an integer list
arr = [1, 2, 3, 4]
# Determine the length of the list
n = len(arr)
# Initialize index at 0
i = 0
# Traverse the list and output each individual element
while i < n: print(arr[i]); i += 1
```

In the given code, it tries to access `arr[n]` which is out of bounds as python list indexing starts from 0 and ends at $n-1$ for a list of length $n$. Also, I used `while` loop as per your request. By initializing `i=0` before the loop and `i += 1` within the loop, we can iterate over the list. This iteration continues until `i` is less than `n`, the length of the list. The `while` loop will not proceed if the list is empty so it correctly handles empty lists.

### G.3 StarCoder Python (Code CPT)

As stated in the main text, this dataset consists of permissively licensed repositories from GitHub, including Git commits, in 80+ programming languages [36]. We chose the Python subset and sub-sampled it to 20B tokens. The full dataset can be found at: https://huggingface.co/datasets/bigcode/starcoderdata

**Example G.4: StarCoder-Python**

```python
"""
function of invoking Gitee API
"""
import base64
import logging
import requests
from flask import current_app
from requests import exceptions
logger = logging.getLogger(__name__)
ORG_URL = "https://gitee.com/api/v5/orgs"
REPO_URL = "https://gitee.com/api/v5/repos"
def get_request(url, params):
    """
    get request
    """
    logger.debug("Get request, connect url: %s", url)
    try:
        response = requests.get(url, params=params)
        return True, response
    except exceptions.ConnectionError as err:
        logger.error(err)
        return False, 'connection error'
    except IOError as err:
        logger.error(err)
        return False, 'IO error'
# more functions truncated...
```

### G.4 OpenWebMath (Math CPT)

As stated in the main text, this dataset contains 14.7B tokens derived from mathematical web pages from Common Crawl, correctly formatted to preserve mathematical content such as LaTeX equations [50]. The dataset can be found at: https://huggingface.co/datasets/open-web-math/open-web-math. As can be seen from the example below, this dataset contains a large amount of English.

**Example G.5: OpenWebMath**

- url: http://math.stackexchange.com/questions/222974/probability-of-getting-2-aces-2-kings-and-1-queen-in-a-five-card-poker-hand-pa
- text: Probability of getting 2 Aces, 2 Kings and 1 Queen in a five card poker hand (Part II). So I reworked my formula in method 1 after getting help with my original question... But I am still getting results that differ...although they are much much closer than before, but I must still be making a mistake somewhere in method 1. Anyone know what it is?

Method 1:
$$P(2A \cap 2K \cap 1Q) = P(Q|2A \cap 2K)P(2A|2K)P(2K)$$
$$= \frac{1}{12}\frac{{4 \choose 2}{46 \choose 1}}{50 \choose 3}\frac{{4 \choose 2}{48 \choose 3}}{52 \choose 5}$$
$$= \frac{(6)(17296)(6)(46)}{(2598960)(19600)(12)}$$
$$= 4.685642 \times 10^{-5}$$

Method 2:
$$\frac{{4 \choose 2} {4 \choose 2}{4 \choose 1}}{52 \choose 5} = \frac{3}{54145}$$
$$5.540678 \times 10^{-5}$$

[Discussion thread follows with comments about conditional probability...]

- date: 2014-03-07 11:01:44

---

## H Theoretical Memory Efficiency Gains with LoRA for Single and Multi-GPU Settings

Modern systems for training neural networks store and operate on the following objects (following the conventions in [51]). Most memory requirements relate to **model states**, which include:

- parameter weights
- gradients
- higher order optimization quantities such as optimizer momentum and variance in the Adam optimizer, and the momentum in the Lion optimizer

The remaining memory requirements come from the **residual states**:

- activations (which depend on batch size and maximum sample sequence length)
- temporary buffers for intermediate quantities in the forward and backward pass

which will require more memory when increasing the batch size and maximum sequence lengths.

LoRA offers memory savings with respect to the model states. The next two sections describe these memory savings in the single GPU and multi-GPU setting with examples loosely inspired by [51].

The data stored at single precision includes:

- a "master copy" of the tuned parameter weights
- the gradient
- all optimizer states (both momentum and variance for Adam, and just momentum for Lion)

For simplicity, we do not consider mixed-precision training, which involves storing critical data at single precision (fp32; 4 bytes per number) while performing some computations at half precision (fp16 or bfloat16; 2 bytes per number).

### H.1 Training on a Single GPU

In the single GPU setup, the difference in memory requirements between LoRA and full finetuning is particularly drastic when using the Adam optimizer [25, 51].

Storing the master weights in fp32 requires 4 bytes per parameter, while storing the gradient in fp32 requires 4 bytes per tuned parameter. In order to maintain the optimizer state in fp32 for Adam, 8 bytes per tuned parameter are required; 4 bytes for the momentum term, and 4 bytes for the variance term. Let $\Psi$ be the number of model parameters. Therefore, in the Adam full finetuning setting of a $\Psi = 7\text{B}$ parameter model, the total memory requirements are at least roughly:

$$4 \times \Psi + 4 \times \Psi + 8 \times \Psi = 112 \text{ GB}$$

The Lion optimizer only uses a momentum term in the gradient calculation, and the variance term in Adam therefore disappears. In the Lion full finetuning setting of a $\Psi = 7\text{B}$ parameter model, the total memory requirements are therefore roughly:

$$4 \times \Psi + 4 \times \Psi + 4 \times \Psi = 84 \text{ GB}$$

LoRA, on the other hand, does not calculate the gradients or maintain optimizer states (momentum and variance terms) for most of the parameters. Therefore the amount of memory used for these terms is drastically reduced.

A LoRA setting with Adam that only tunes matrices that are 1% of the total parameter count (e.g. $\Psi = 7\text{B}$ base model with 70M additional parameters used by LoRA) requires roughly $4 \times \Psi (1 + 0.01) + 4 \times \Psi \times 0.01 + 8 \times \Psi \times 0.01 = 29.12$ GB of memory. Theoretically this can be reduced further to $2 \times \Psi + 16 \times \Psi \times 0.01 = 15.12$ GB if the non-tuned parameter weights are stored in bfloat16. We use this assumption for the subsequent examples.

Note again that these numbers do not take into consideration sample batch size or sequence length, which affect the memory requirements of the activations.

**Table S14: Theoretical memory required to store the model and optimizer state during training for a 7B parameter model.**

| 7B Training | 1 GPU | 8 GPUs | 16 GPUs | 32 GPUs | 64 GPUs |
|---|---|---|---|---|---|
| Adam | 112 GB | 14 GB | 7 GB | 3.5 GB | 1.75 GB |
| Adam + LoRA | 15.12 GB | 1.89 GB | 0.945 GB | 0.4725 GB | 0.236 GB |
| Lion | 84 GB | 10.5 GB | 5.25 GB | 2.625 GB | 1.3125 GB |
| Lion + LoRA | 14.84 GB | 1.855 GB | 0.9275 GB | 0.464 GB | 0.232 GB |

Note that the numbers exclude memory needed to store activations. FSDP sharding the parameter and optimizer states across $N$ devices results in less memory usage relative to LoRA. LoRA on the other hand enables training on GPUs with far less memory and also enables training without needing as many GPUs to shard across.

### H.2 Training on Multiple GPUs with Fully Sharded Data Parallelism

Past approaches for training LLMs across multiple GPUs include model parallelism, where different layers of the LLM are stored on different GPUs. However this requires high communication overhead and has very poor throughput [51]. **Fully Sharded Data Parallelism (FSDP)** shards the parameters, the gradient, and the optimizer states across GPUs. This is incredibly efficient and is actually competitive with the memory savings offered by LoRA in certain settings.

FSDP sharding of the parameter and optimizer states across $N$ devices results in less memory usage relative to LoRA. LoRA on the other hand enables training on GPUs with far less memory and also enables training on fewer GPUs.

For example, in the Adam full finetuning setting of a $\Psi = 7\text{B}$ parameter model on 8 GPUs with FSDP, the total memory requirement for each GPU is roughly $(4 \times \Psi + 4 \times \Psi + 8 \times \Psi) / 8 = 14$ GB. This reduces further to 3.5 GB for FSDP with 32 GPUs (see Table S14).

The LoRA with Adam setup on 8 GPUs (where $\Psi = 7\text{B}$ base model and there are 70M additional LoRA parameters) requires roughly $(2 \times \Psi + 16 \times \Psi \times 0.01) / 8 = 1.89$ GB of memory per GPU. With 32 GPUs this decreases further to 0.4725 GB.

Standard industry level GPUs have on-device memory between 16 GB (e.g. V100s) and 80 GB (e.g. A100s and H100s). As Table S14 demonstrates, the per-GPU memory requirements for training a 7B parameter model decrease drastically as the number of GPUs increases. The memory requirements for training a 7B model with Adam + LoRA on a single GPU are 15.12 GB, but the same per-GPU memory requirement for training a 7B model with Adam but without LoRA on 8 GPUs is 14 GB. In this 8 GPU scenario, the efficiency gains from LoRA disappear.

Table S15 applies similar calculations to a 70B parameter model. Finetuning such a large model on 8 GPUs is only possible using a technique like LoRA; where Adam requires 140 GB per GPU, Adam+LoRA requires 18.9 GB per GPU. The efficiency gains of LoRA relative to FSDP therefore depend on the model size and GPU availability/cost considerations.

We do the same analysis for a 405B parameter model to highlight how LoRA is beneficial as model size scales (Table S16). This is particularly relevant now that Llama-3-405B has been released by Meta [13].

**Table S15: Theoretical memory required to store the model and optimizer state during training for a 70B parameter model.**

| 70B Training | 1 GPU | 8 GPUs | 16 GPUs | 32 GPUs | 64 GPUs |
|---|---|---|---|---|---|
| Adam | 1.12 TB | 140 GB | 70 GB | 35 GB | 17.5 GB |
| Adam + LoRA | 151.2 GB | 18.9 GB | 9.45 GB | 4.725 GB | 2.36 GB |
| Lion | 840 GB | 105 GB | 52.5 GB | 26.25 GB | 13.125 GB |
| Lion + LoRA | 148.4 GB | 18.55 GB | 9.275 GB | 4.64 GB | 2.32 GB |

**Table S16: Theoretical memory required to store the model and optimizer state during training for a 405B parameter model. Units are in gigabytes (GB).**

| 405B Training | 1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| Adam | 6480 | 810 | 405 | 202.5 | 101.25 | 50.625 | 25.3 |
| Adam + LoRA | 874.8 | 109.35 | 54.65 | 27.34 | 13.67 | 6.83 | 3.42 |
| Lion | 4860 | 607.5 | 303.75 | 151.875 | 75.94 | 37.97 | 18.98 |
| Lion + LoRA | 858.6 | 107.325 | 53.66 | 26.83 | 13.42 | 6.71 | 3.35 |

---

## I LoRA Throughput and Memory Measurements

We report training efficiency comparisons between full finetuning and models trained with LoRA for various choices of rank. We measured both the **throughput** (in tokens per second) and **peak active memory** (in GB) for training runs representative of the experiments reported in the paper. We performed the runs using a single node of 8×H100-80GB GPUs. We used a per-GPU micro batch size of 1 and targeted all linear layer weights with LoRA (i.e. both Attention and MLP).

In Figure S10 we observe that there is a significant gap between full finetuning and LoRA runs, related to the additional overheads of the LoRA computations. In general, **LoRA leads to an approximately 15% reduction in throughput** for a given batch size. LoRA with higher ranks is slower than lower ranks across all batch sizes; this is particularly noticeable for rank $r = 512$. Similarly, LoRA settings with higher batch sizes have slightly higher throughput relative to lower batch sizes. Some of the slowdown is intrinsically related to the overheads of performing LoRA, since in practice it involves more computations of intermediate activations. However, we note that we did not optimize the LoRA implementation and used the publicly available HuggingFace peft library, which might be amenable to further optimizations that could reduce the gap in throughput.

For peak memory, we notice that for small batch sizes, LoRA provides a substantial reduction in peak memory (∼ 40%). This is expected since the optimizer state is significantly smaller when using parameter efficient methods. However, as batch size increases, the size of intermediate activations increases proportionally, dominating the required memory. We limit the per GPU micro batch size to 8 to prevent out of memory errors, so for batch sizes 64 and above, we perform gradient accumulation. This leads to the throughput and memory stabilizing for batch size 64 and above, with just around (∼ 15% memory savings) for larger batch sizes.
