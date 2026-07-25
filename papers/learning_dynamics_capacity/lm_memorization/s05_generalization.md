## 4. Disentangling Unintended Memorization from Generalization

The paper next replaces independent synthetic strings with real text, where generalization is possible. Restricting the reference model’s computation connects its predictions to $\mathcal{V}$-information [53], the information usable under a particular model class.

### Experimental details

The authors repeat the synthetic-data design on `FineWeb` [40], selecting 64-token sequences and performing an extra deduplication pass. Without this pass, truncation would leave roughly 1–2% duplicate sequences, which would corrupt extraction measurements. They pretrain varied model sizes on varied corpus sizes, then measure unintended memorization for each model/dataset pair. They also evaluate standard loss-based membership inference and exact extraction by greedy decoding from prefixes of different lengths.

### Results

At the sample level, unintended memorization grows with parameter count and falls as the training set grows. Relative to an oracle reference, a smaller target model initially learns training examples better than the oracle and its memorization increases. Once the target starts capturing patterns that transfer beyond its own dataset, it performs worse on average than the high-capacity oracle and the measured unintended component decreases.

### Dataset-to-capacity ratio predicts double descent

For large datasets, evaluation loss begins to decrease only after the model reaches its memorization capacity—around $10^5$ samples in the studied regime, depending on parameter count. The authors use the ratio between dataset information size and estimated model capacity, rather than only counts of samples or parameters. Under this calibration, **double descent begins exactly when data capacity exceeds model capacity** [5, 38]: storing additional particulars is no longer beneficial, so learning shifts toward generalizable structure.

### Generalization explains nonzero extraction rates

The work also observes nonzero extraction on held-out text. Its interpretation is that successful extraction in this setting is attributable to **generalization**, not necessarily to sample-specific memorization. This distinction is why raw extraction and likelihood alone cannot answer how much a model retained about a particular training dataset.
