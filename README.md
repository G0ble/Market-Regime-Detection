# Market Regime Detection: HMM vs Wasserstein Clustering

Two approaches to detecting market regimes — a Gaussian Hidden Markov Model and Wasserstein
k-means clustering (Horvath, Issa & Muguruza, 2021) — implemented with fully causal signals
and evaluated against non-trivial baselines.

The emphasis throughout is on *evaluation honesty*: both models are built so that every signal
could actually have been traded at the time, and both are measured against a one-line rule on
realised volatility that they have to beat to justify their complexity.

---

## Headline result

**The rule translating signal into position mattered more than the choice of model.**

The identical HMM signal, with no change to the model, produces:

| Exposure rule | Sharpe |
|---|---|
| Continuous — `exposure = 1 − P(stress)` | **0.69** |
| Binary — `exposure = 0.5 if stress else 1.0` | **0.79** |

Ten points of Sharpe from the translation rule alone. Under the continuous rule the HMM *loses*
to the volatility baseline; under the binary rule it beats it. The question "does regime
detection add value?" has no answer independent of how the signal becomes a position — and this
is rarely tested.

## Full results

Net of 5 bps transaction costs on turnover. Walk-forward, out-of-sample, 2010–2024, SPY.

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| Buy & Hold | 12.25% | 17.11% | 0.72 | −35.75% | — |
| Realised-vol targeting *(baseline)* | 8.03% | 10.85% | 0.74 | −19.75% | 3.5× |
| HMM — continuous exposure | 7.07% | 10.24% | 0.69 | −22.39% | 4.2× |
| HMM — binary exposure | 9.76% | 12.39% | 0.79 | −21.34% | — |
| **Wasserstein — binary exposure** | **10.23%** | **12.24%** | **0.84** | **−19.98%** | 3.5× |

WK-means edges out the HMM (0.84 vs 0.79), consistent with its non-parametric nature — but
**0.05 of Sharpe over 15 years of one asset is not statistically decisive**. The honest reading
is "scored slightly higher in this experiment", not "is superior".

---

### Leak 1 — smoothed probabilities

`hmmlearn.predict_proba()` returns **smoothed** probabilities `P(state_t | obs_1..T)`, which
condition on the *entire* sample. `predict()` uses Viterbi, also a global decode. Neither is
tradeable. The **forward algorithm** gives filtered probabilities `P(state_t | obs_1..t)` — its
recursion only ever touches `t−1` and `t`.

Measured bias: **mean 3.5%, but up to 82% at regime transitions** — exactly where a strategy
makes its decisions. The spikes run in both directions: smoothed both *anticipates* real crises
and *retroactively suppresses* false alarms that a live signal would have acted on.

### Leak 2 — the model itself 

Filtered probabilities fix the inference, but a model **fitted on the full sample** still knows
what a crisis looks like because it has seen COVID. Closing this requires walk-forward: refit on
a rolling 4-year window, monthly, and assign forward with the frozen model.

Measured impact: mean difference of **~0.11** between the in-sample and walk-forward signals —
roughly **three times** the smoothing bias. Most tutorials that address look-ahead at all address
only Leak 1.

### Ancillary requirements

- **Label switching** — EM has no commitment to state numbering; "state 0" can be calm in one
  refit and stressed in the next. States are anchored by sorting on variance. Without this the
  signal would invert at random across refits.
- **Expanding-median baseline** — even the *baseline* must be causal. A full-sample median would
  mean the 2010 threshold is informed by 2020 volatility.
- **One-day lag on exposure** — a signal from today's close cannot trade today's return.

---

## On the Wasserstein implementation

In 1D, optimal transport has a closed form: sort both samples, pair the quantiles, average the
p-th powers of the gaps. The barycenter likewise — the average of the quantile functions.


**A finding worth flagging.** Normalising each window by its own standard deviation — stripping
magnitude and leaving only *shape*, supposedly the non-parametric advantage — collapses the
clustering to ~48/52, essentially noise. All of WK-means' discriminative power on this data comes
from **magnitude**, the same signal the HMM uses. The flexibility to see the tails did not, here,
find structure the moments missed.

Separately, K=2 WK-means degenerates into a **crash detector**: the minority cluster fires only
~4% of the time, exclusively in 2008/2009/2011/2020, classifying the 2015-16 correction, 2018
selloff and the entire 2022 bear market as "calm". K=3 recovers a usable intermediate state.

---


## Caveats

Stated so the reader can calibrate how much model selection went into the winning number.

- **Variants explored:** K ∈ {2,3,4} for both methods; p ∈ {1,2} for Wasserstein; normalised vs
  raw windows; three feature sets; two exposure rules. The exploration was diagnostic-driven
  rather than a blind sweep, but each additional variant raises the chance the winner won by
  luck.
- **A failed hypothesis, reported:** conditioning the HMM on negative momentum — motivated by the
  observation that it de-risks through volatile *rallies* — **made things worse** (Sharpe 0.62,
  drawdown −28%). A momentum rule *alone* scored 0.74, meaning the HMM actively subtracted value
  in that combination.
- **A 3-state HMM degrades under walk-forward.** The meaning of the states drifts: "crisis"
  represents ~17% volatility in 2017 and ~65% in 2021. In extended bull markets the training
  window contains no real stress and the model subdivides noise. Regimes detected by a rolling
  HMM are **relative to the training window, not absolute** — the unavoidable cost of choosing a
  rolling window over an expanding one.
- **Single asset, single period.** No cross-sectional or out-of-period validation.
- **Overlapping windows.** With step=1, adjacent Wasserstein windows share 20 of 21 observations.
- **Flat 5 bps costs.** No market-impact or slippage modelling.
- `hmmlearn._compute_log_likelihood` is a private API and may break across versions.

## Reference

Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein
Distance.* [arXiv:2110.11848](https://arxiv.org/abs/2110.11848) — published in the *Journal of
Computational Finance*, 28(1), 2024.
