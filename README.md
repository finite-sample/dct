# Making Neural Networks Stable with Dropout‑Consistency Training

Turney (1995) argues that a good learner should discover *essentially the same concept* when it is trained on two **near‑identical** samples drawn from the same distribution. Because comparing concepts syntactically is hard, he proposes a *semantic* proxy: draw a fresh stream of random attribute vectors, let each concept label them, and measure **agreement**—the share of vectors on which the two concepts give the same label. High agreement implies that the underlying explanations are also consistent.

We borrow that idea but flip the workflow. Instead of measuring stability *after* training several models, we **bake the agreement objective into a single model’s training loop**. During each minibatch we forward the data through the network multiple times under independent dropout masks, average the usual cross‑entropy, and penalise disagreement among the resulting probability distributions. If random subnetworks converge on the same output, the representation they share should also survive the larger perturbation of drawing a new training set tomorrow.

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
2. **Dual training sets** The remaining 70 % is split into two equal halves. Each half trains a separate network—e.g. *method A* (standard) versus *method D* (dropout‑consistency)—using identical hyper‑parameters but independent random seeds.
3. **Prediction collection** With dropout **disabled** (`model.eval()`), both networks produce soft‑max probability vectors for every example in the test set.
4. **Agreement score** For each test example we compute the symmetric KL divergence
   $\text{SKL}(P,Q)=\tfrac12\bigl[\operatorname{KL}(P\!\parallel\!Q)+\operatorname{KL}(Q\!\parallel\!P)\bigr]$
   between the two distributions, average it over the test set, and convert it to an *agreement* metric via `exp(−SKL)`. Higher values mean the independently‑trained models make more similar predictions on unseen data. (We also log an MSE‑based score for completeness.)
5. **Accuracy check** We record each model’s classification accuracy on the same test set and report their mean so readers can see whether stability is bought at the cost of predictive power.

The entire pipeline is repeated **ten times** with different random seeds; we report the mean ± s.d. of both stability and accuracy. Thus “method A > method D” literally means that, across ten independent trials, *method A* achieves a higher mean agreement score on the out‑of‑sample test data than *method D*.

---

## Relationship to R‑Drop

**R‑Drop** feeds each training example through *two* dropout masks and adds a **bidirectional KL** term to the loss to narrow the train–test gap of a *single* run. Our method differs in three ways:

1. **Scope** – We can average over *n ≥ 2* passes, not just two.
2. **Penalty** – We use mean‑squared error for efficiency; any distance could work.
3. **Goal** – We judge success by **between‑bootstrap agreement**, the quantity Turney links to conceptual repeatability, rather than by the validation accuracy of one model.

---

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
