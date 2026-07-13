## D. Editing Different Categories of Facts Together

For an edit $(s,r,o)$, relation $r$ links subject $s$ and object $o$. Subjects and objects have associated types $\tau(s)$ and $\tau(o)$. For example, $r=\text{“is a citizen of”}$ links a `Person` to a `Country`. Two subjects are considered diverse if $\tau(s_1)\neq\tau(s_2)$, and similarly for objects.

For each relation pair $(r_1,r_2)$, the authors sample a mixed edit set $\mathcal{E}_{mix}=\{(s,r,o)\mid r\in\{r_1,r_2\}\}$ with equal numbers of edits per relation. They compare MEMIT in four diversity regimes:

1. **Different subjects, different objects**:
   - `Person` — citizen of (`P27`) → `Country`
   - `Country` — official language (`P37`) → `Language`
2. **Similar subjects, different objects**:
   - `Person` — plays position in sport (`P413`) → `Sport position`
   - `Person` — native language (`P1412`) → `Language`
3. **Different subjects, similar objects**:
   - `Place` — located in (`P17`) → `Country`
   - `Item/Product` — country of origin (`P495`) → `Country`
4. **Similar subjects, similar objects**:
   - `Person` — citizen of (`P27`) → `Country`
   - `Person` — works in (`P937`) → `City/Country`

Figure D shows that MEMIT’s rewrite performance in mixed settings closely follows the average of the individual relation splits. The presence or absence of diversity in the edits does not tangibly change performance.

## E. Demonstrations

The paper gives two case studies applying MEMIT to bulk-edit new or corrected memories into `GPT-J` (6B).

**Knowledge freshness.** The authors update `GPT-J` with newly available factual results from a large public event in November 2022, representing updates as `(congressperson, elected from, district)` and `(governor/senator, elected from, state)`. MEMIT achieves **100% efficacy (ES)** and **94% generalization (PS)** on this edit set.

**Specialized knowledge domain.** The authors use MEMIT to create a model with amateur astronomy knowledge. They scrape stars referenced more than 100 times in `Wikidata` and belonging to one of 18 constellations: `Andromeda`, `Aquarius`, `Cancer`, `Cassiopeia`, `Gemini`, `Hercules`, `Hydra`, `Indus`, `Leo`, `Libra`, `Orion`, `Pegasus`, `Perseus`, `Pisces`, `Sagittarius`, `Ursa Major`, `Ursa Minor`, and `Virgo`. This yields **289** tuples of the form `(star, belongs to, constellation)`. Unmodified `GPT-J` recalls the constellation of a star with **53%** accuracy; after MEMIT, accuracy rises to **86%**.

## F. Ablations

MEMIT relies on several critical choices: using a range of mid-layer causal MLPs, targeting MLP modules rather than attention, editing at the last subject token, and setting the covariance hyperparameter $\lambda$. The last-subject-token choice had already been studied by ROME, so the paper ablates the other factors.

### F.1. Varying the Number and Location of Edited Layers

The authors test five configurations of $\mathcal{R}$. Four lie in the high-causal-effect region identified by causal tracing, and one lies in late MLP layers with low causal effect. Figure 11 shows that using more critical layers improves efficacy, generalization, and specificity. Editing late MLPs performs much worse, confirming that MEMIT benefits from causal localization.

### F.2. Varying the Targeted Module: Editing Attention

The authors test whether editing early or late attention modules can match MLP edits. Figure 12 shows that attention edits perform considerably worse, reinforcing the claim that MLPs are the appropriate intervention target for factual-memory editing.

### F.3. Varying the Covariance Hyperparameter $\lambda$

The covariance adjustment $\lambda$ controls how strongly the update preserves pre-existing associations. Figure 13 shows that specificity and fluency increase monotonically with $\lambda$, while efficacy and generalization fall as $\lambda$ increases. The aggregated score peaks around $\lambda\approx10^4$, matching the chosen `GPT-J` value of $15000$.

### `COUNTERFACT` sample

The appendix includes a concrete `COUNTERFACT` case. The edit changes `Percy Snow` under relation `P413` (“plays position in sport”) from true target `linebacker` to new target `goaltender`. Paraphrase prompts test whether the new answer generalizes, neighborhood prompts use other football players to test specificity, and generation prompts test free-form consistency.
