## 5. Memorization and Membership

The controlled setting fixes train/test data and eliminates duplicates, making it well suited to study how model size, dataset size, and membership-inference success interact.

### 5.1. Membership in synthetic and text data

For models trained on synthetic data, the authors run a loss-based membership attack [54, 44] across dataset sizes. Once a dataset becomes sufficiently large relative to the model, average-case membership inference begins to fail. Thus even when a model was trained on a point, recognizing an average training sample need not be possible if its capacity is spread across too much data.

The same methodology is applied to text, with strict deduplication and matched held-out samples. The central comparison is capacity-relative rather than merely parameter-relative: text is heterogeneous and generalization overlaps training and test loss distributions much more than independent random strings do.

### 5.2. Scaling laws for membership

The authors fit predictive models of the F1 score of a loss-based membership attack from token count, number of examples, and parameter count, then validate them on models from $500\mathrm{K}$ to $1.5\mathrm{B}$ parameters.

#### 5.2.1. Functional forms

At fixed capacity, membership F1 is approximately sigmoidal in dataset size. A large model trained on a tiny dataset can be strongly overfit, producing F1 near 1; as the data grows, train and test losses become harder to separate and F1 approaches 0.5. The fitted functional form is

$$
\operatorname{Membership}_{F_1}(\theta,\mathcal{D})
=\frac{1}{2}\left(1+c_1\,\sigma\!\left(c_2\left(\frac{\operatorname{Capacity}(\theta)}{|\mathcal{D}|}+c_3\right)\right)\right),
$$

where $\sigma$ is the logistic sigmoid and $c_1,c_2,c_3$ are fitted constants. In the limit $|\mathcal{D}|\to\infty$, the prediction is F1 $\to0.5$, so both average-case loss-based membership inference and extraction become impossible within this formulation.

#### 5.2.2. Validation on larger models

The paper validates the scaling law using `GPT2-Medium` (123.7M parameters) and `GPT2-XL` (1.56B parameters), selecting dataset sizes that should yield F1 values 0.55, 0.75, and 0.95. The predicted and observed results are:

| Model | Parameters | Dataset size | Predicted F1 | Observed F1 |
|---|---:|---:|---:|---:|
| `GPT2-XL` | 1,556,075,200 | 170,654,583 | 0.55 | **54.61 ± 1.3** |
| `GPT2-XL` | 1,556,075,200 | 76,795,021 | 0.75 | 71.08 ± 0.4 |
| `GPT2-XL` | 1,556,075,200 | 18,851,574 | 0.95 | **95.85 ± 0.8** |
| `GPT2-Medium` | 123,702,528 | 13,566,442 | 0.55 | **53.44 ± 1.1** |
| `GPT2-Medium` | 123,702,528 | 6,104,935 | 0.75 | 65.69 ± 0.6 |
| `GPT2-Medium` | 123,702,528 | 1,498,634 | 0.95 | **97.98 ± 0.3** |

Contemporary LLMs typically use at least $10^2$ tokens per parameter. The fitted law predicts F1 of about 0.5 in that regime, implying no statistically significant average-case loss-based membership inference under this measure. This conclusion concerns average points and does not negate privacy risks for atypical or unusually memorized data.
