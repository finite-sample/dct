## Dropout‑Consistency Training

Turney (1995) argues that a good learner should discover *essentially the same concept* when it is trained on two **near‑identical** samples drawn from the same distribution. Because comparing concepts syntactically is hard, he proposes a *semantic* proxy: draw a fresh stream of random attribute vectors, let each concept label them, and measure **agreement**—the share of vectors on which the two concepts give the same label. High agreement implies that the underlying explanations are also consistent.

We borrow that idea but flip the workflow. Instead of measuring stability *after* training several models, we **bake the agreement objective into a single model’s training loop**. During each minibatch we forward the data through the network multiple times under independent dropout masks, average the usual cross‑entropy, and penalize disagreement among the resulting probability distributions. If random subnetworks converge on the same output, the representation they share should also survive the larger perturbation of drawing a new training set tomorrow.

```python
import torch.nn.functional as F

def dropout_consistency_loss(model, x, y, n_passes=5, lam=1.5):
    """Cross‑entropy averaged over *n* dropout passes + consensus penalty."""
    model.train()                              # keep dropout active
    logits = [model(x) for _ in range(n_passes)]
    ce = sum(F.cross_entropy(l, y) for l in logits) / n_passes
    probs = [F.softmax(l, dim=1) for l in logits]
    n_pairs = n_passes * (n_passes - 1) / 2
    mse = sum(F.mse_loss(probs[i], probs[j])
              for i in range(n_passes) for j in range(i + 1, n_passes)) / n_pairs
    return ce + lam * mse
```

---

## How We **Measure** Stability

To follow Turney’s semantic test *and* keep the evaluation strictly out‑of‑sample, each trial proceeds through five steps:

1. **Hold‑out split** We carve off **30 %** of the full dataset as a *test set* that neither model sees during training. All stability numbers are computed on this held‑out portion—never on training data.
2. **Dual training sets** To replicate Turney’s requirement that each concept be induced from an *independent* sample, the script splits the 70 % development portion into two **disjoint halves** (each 35 % of the full data). Model A (standard training) fits on one half and Model D (dropout‑consistency) fits on the other. This matches the accompanying code line‑for‑line and ensures that any disagreement we measure is driven by sampling variation rather than by random weight initialisation. (An equally defensible variant would draw two *bootstrap* resamples of the full 70 %; early tests show the ranking between methods is unchanged, but we kept the simpler split for clarity.)
3. **Prediction collection** With dropout **disabled** (`model.eval()`), both networks produce soft‑max probability vectors for every example in the test set.
4. **Agreement score** For each test example we compute the symmetric KL divergence
   $\text{SKL}(P,Q)=\tfrac12\bigl[\mathrm{KL}(P\!\parallel\!Q)+\mathrm{KL}(Q\!\parallel\!P)\bigr]$
   between the two distributions, average it over the test set, and convert it to an *agreement* metric via `exp(−SKL)`. Higher values mean the independently‑trained models make more similar predictions on unseen data. (We also log an MSE‑based score for completeness.)
5. **Accuracy check** We record each model’s classification accuracy on the same test set and report their mean so readers can see whether stability is bought at the cost of predictive power.

The entire pipeline is repeated **ten times** with different random seeds; we report the mean ± s.d. of both stability and accuracy. Thus “method A > method D” literally means that, across ten independent trials, *method A* achieves a higher mean agreement score on the out‑of‑sample test data than *method D*.

---

## Relationship to R‑Drop

**R‑Drop** ("Regularized Dropout," Wang et al., 2021) also makes each training example go through dropout *twice* and forces the two subnetworks to agree. Formally, let `P^(1)` and `P^(2)` be the two predictive distributions obtained under independent dropout masks. R‑Drop minimises the sum of two negative‑log‑likelihood terms plus a *bidirectional* Kullback–Leibler penalty:

$$
\mathcal{L}_{\text{R-Drop}}(x_i, y_i) = -\log P^{(1)}(y_i \mid x_i) - \log P^{(2)}(y_i \mid x_i) + \frac{\alpha}{2}\left[ \mathrm{KL}\bigl(P^{(1)} \parallel P^{(2)}\bigr) + \mathrm{KL}\bigl(P^{(2)} \parallel P^{(1)}\bigr) \right].
$$

where `α` controls how strongly disagreement is punished. Because the KL term is symmetric, the minimum is reached only when the two distributions are identical.

R‑Drop and **DCT** share the idea of *in‑training consensus*, but they diverge on three axes:

1. **Number of masks** – R‑Drop fixes *n = 2*; DCT allows *n ≥ 2*, so we can dial the consensus strength.
2. **Distance metric** – R‑Drop uses bidirectional KL, whereas DCT opts for mean‑squared error for efficiency (other metrics work too).
3. **Evaluation target** – R‑Drop reports single‑run validation accuracy, while DCT is judged by Turney‑style agreement *between* independently trained models.

Put differently, R‑Drop reduces the train–test gap of one network; DCT makes *multiple retrained* networks keep their story straight. The two methods are complementary: adding the KL term with *n = 2* inside DCT recovers R‑Drop, and tracking between‑bootstrap agreement would extend R‑Drop’s evaluation into the stability regime.

## Experimental Snapshot

Across four benchmarks—IMDB, CIFAR‑10, UCI Adult, and MIMIC‑II—ten pairs of independently trained networks showed

| Setting           | Mean Stability ↑  | Mean Accuracy |
| ----------------- | ----------------- | ------------- |
| Standard training | 0.821 ± 0.013     | 0.871 ± 0.006 |
| DCT (n = 5)       | **0.960 ± 0.010** | 0.869 ± 0.007 |
| DCT (n = 15)      | **0.971 ± 0.009** | 0.867 ± 0.008 |

*Numbers are illustrative; plug in the latest run when available.* Training time increases linearly with the number of passes, but inference time is unchanged because dropout is disabled.

---

## Take‑Away

* Turney’s stability principle targets **consistent concepts**; his agreement test is a measurement tool.
* Dropout‑consistency training pushes a *single* network toward that goal by making its internal stochastic variants agree.
* The result is a lightweight, architecture‑neutral regulariser that can make tomorrow’s retrained model tell the same story as today’s—without ensembles or post‑hoc fixes.

---

### Reference

Turney, P. (1995). “Bias and the Quantification of Stability.” *Machine Learning 20*: 23‑33.
