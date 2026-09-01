# FlareFinder — Concepts, Numbers, and How to Read Them

Written 2026-09-01. This is the file to read before you write a sentence of
the paper, and the file to re-read if a number in the output confuses you.

Everything here is **measured on this project's own data**, not copied from a
textbook. Where a number is quoted, the command that produces it is given.

---

## 1. THE ONE-PARAGRAPH VERSION

Gas flares burn off unwanted gas at oil sites. The official global record
(VIIRS Nightfire, published by EOG) misses small flares. We are building a
system that finds those missing flares using a second satellite (Sentinel-2)
and estimates how much gas they burn. The hard parts are not the machine
learning — they are (a) not fooling ourselves with leaky test sets, (b) not
mistaking hot desert for a flare, and (c) being honest that our "ground
truth" comes from the very sensor whose failures we are studying.

---

## 2. UNITS — get these wrong and every number is wrong

| Unit | Meaning | Conversion |
|---|---|---|
| **BCM** | Billion cubic metres per year. What EOG publishes. | 1 BCM = 1,000 Mm³ |
| **Mm³/yr** | Million cubic metres per year. `volume_mcm` in our tables. | |
| **m³/h** | Cubic metres per hour. The unit thresholds are quoted in. | 1 Mm³/yr ÷ 8,760 h ≈ 114.2 m³/h |

`src/config.py` holds `HOURS_PER_YEAR = 8760.0` and the conversion helper.
**Never re-derive the conversion in another module** — that is how two parts
of a pipeline start disagreeing.

### The Seymour threshold
Seymour et al. 2025 found VIIRS misses flares below **~360 m³/h**. Converted:

```
360 m³/h × 8,760 h = 3,153,600 m³/yr = 3.1536 Mm³/yr
```

**The assumption inside that conversion matters.** It assumes the flare burns
continuously all year. Most do not — see §6. So 3.1536 Mm³/yr is an *upper
bound* on the annual equivalent, and counting sites below it is a
**conservative undercount** of the small-flare population.

---

## 3. SIZE BINS — why every result is split by flare size

**RULE 2** says error must be reported as a function of flare size. A single
pooled error number is dominated by the biggest flares, which are exactly the
ones this project is *not* about.

Bin edges (`config.yaml → evaluation.size_bins_m3_per_h`), in m³/h:

```
0 ──── 100 ──── 360 ──── 1,000 ──── 5,000 ──── 20,000 ──── ∞
  tiny   sub-VIIRS  small    medium     large     very-large
```

The **360 edge is the Seymour detection limit**. So `tiny` and `sub-VIIRS`
together are the sub-threshold regime — the population your project exists to
find. That is why the bins are cut there and not at round numbers.

A bin is only reported if it has at least **20 test sites**
(`min_sites_per_bin`). Below that, a metric is noise wearing a number's
clothes. The module prints `usable: False` instead of a value.

---

## 4. READING THE BASELINE TABLE

This is the output of `python run.py baseline`:

| size_bin | n | median_log_bias | ci_lo | ci_hi | bias_factor | excludes_zero |
|---|---|---|---|---|---|---|
| tiny | 292 | +0.0876 | 0.0564 | 0.1514 | 1.223 | True |
| sub-VIIRS | 355 | −0.0391 | −0.0790 | 0.0258 | 0.914 | **False** |
| small | 244 | +0.2251 | 0.1304 | 0.2647 | 1.679 | True |
| medium | 347 | +0.2344 | 0.1927 | 0.2748 | 1.715 | True |
| large | 275 | −0.1982 | −0.2321 | −0.1580 | 0.634 | True |
| very-large | 139 | −0.6862 | −0.7240 | −0.6486 | 0.206 | True |

### `median_log_bias` — read this as "how many powers of ten wrong"
It is `log₁₀(predicted) − log₁₀(actual)`, the median across sites in that bin.

- **Positive → the model OVERESTIMATES.**
- **Negative → the model UNDERESTIMATES.**
- **Zero → unbiased.**

The unit is sometimes called a **dex** (decimal exponent). +1 dex = 10× too
high. +0.3 dex ≈ 2× too high.

**Why log space and not plain error?** An error of 100 m³/h is catastrophic on
a 50 m³/h flare and irrelevant on a 50,000 m³/h one. A plain (linear)
residual would be dominated by the largest flares and would tell you nothing
about the small ones. Log space asks "off by what *factor*", which is the
question that means the same thing at every size.

### `bias_factor` — the same number, in plain language
`bias_factor = 10^(median_log_bias)`. It is there because "×1.72 too high"
is legible and "+0.234 dex" is not. Use the factor in the paper's prose and
the log bias in the tables.

- `1.715` → predicts **1.7× too much**
- `0.206` → predicts **0.206× the truth**, i.e. **~4.9× too little**

### `ci_lo` / `ci_hi` — the 95% confidence interval
**RULE 3**: every headline number carries one. Produced by **bootstrap**:

> Resample the test sites *with replacement* 10,000 times, recompute the
> median bias each time, and take the 2.5th and 97.5th percentiles of those
> 10,000 answers.

It answers: *if I had drawn a different sample of flares, how much would this
number have moved?* A wide interval means "we do not really know."

Bootstrap needs no assumption that the data is bell-shaped, which matters
here because flare volumes are anything but.

Seed: `seeds.bootstrap = 2718`, so the interval is identical on every rerun.

### `excludes_zero` — the honesty column
`True` means the confidence interval does not contain zero, so the bias is
distinguishable from "no bias at all."

**`sub-VIIRS` is False.** Its interval runs −0.079 to +0.026, straddling zero.
So: *we cannot demonstrate bias in the sub-VIIRS bin.* Do not claim one. That
one `False` is more scientifically valuable than the five `True`s, because it
shows the analysis is capable of returning "no."

### The headline
```
small bins  (tiny + sub-VIIRS)  mean log bias  +0.024  (×1.06)
large bins  (large + very-large)               −0.442  (×0.36)
size-dependent gap                             +0.466 dex
```
Small flares are overestimated relative to large ones by about **0.47 dex, a
factor of ~2.9**. Direction matches Elvidge et al. 2024 (small over, large
under) — arrived at independently on Gulf data.

### ⚠️ THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS
This is **not** a reproduction of EOG's calibration:

- `corr(log temperature, log volume) = +0.025` — temperature carries nothing
- `corr(log ellipticity, log volume) = −0.037` — nothing
- `detection_freq` **alone** gives R² = 0.831; the other two add **+0.007**
- The fitted temperature exponent is **−2.81**; Stefan-Boltzmann says ~**+4**.
  That negative value is a *suppression artefact*, not physics.

What is actually fitted is a **detection-frequency regression**. EOG's real
formula multiplies **radiant heat** by a fitted slope (`0.029353`, visible in
the workbook filenames), and neither radiant heat nor source area exists in
the per-site sheets.

**Sayable:** "A calibration built from the observables in the public EOG
catalogue has strongly size-dependent error."
**Not sayable:** "We reproduced the standard VIIRS calibration and showed it
is biased."

---

## 5. HOLDOUTS AND THE KS STATISTIC

### Why holdouts (RULE 1)
Test on data the model has never seen, or the score is a memory test. We
enforce three kinds, and `_assert_disjoint()` in `src/splits.py` **raises an
exception** rather than warning — a silent leak is worse than a crash.

| Split | What is held out | The claim it supports |
|---|---|---|
| **by-site** | 20% of sites, all their years | **Headline performance** |
| **by-year** | 2023 + 2024 | Forward monitoring |
| **by-region (LORO)** | one country at a time | Transfer across geography |

### The KS statistic — "are train and test even comparable?"
**Kolmogorov–Smirnov**: take the distribution of log volume in train and in
test, and measure the **largest vertical gap between their cumulative
curves**. It runs 0 → 1.

- **0** = identical distributions
- **1** = no overlap at all
- **Our threshold: 0.150** (`max_acceptable_ks`)

**Why it matters here.** If you hold out Saudi Arabia and the model scores
badly, there are two explanations: the model doesn't travel between countries,
*or* the model is bad at small flares and Saudi flares are small. A pooled
number cannot tell you which. KS measures how badly those two are tangled.

Current values (`python run.py splits`):

| Split | test n | KS raw | KS after matching | Verdict |
|---|---|---|---|---|
| by-site | 2,355 | **0.045** | 0.016 | **CLEAN** |
| by-year | 1,647 | 0.209 | 0.049 | confounded |
| region −IRN | 4,487 | 0.195 | 0.020 | confounded |
| region −IRQ | 2,656 | 0.264 | 0.035 | confounded |
| region −SAU | 4,644 | **0.362** | 0.044 | confounded, very-large unusable |

**by-site is the only clean split.** It is the only one that can carry a bare
"MAE = X ± Y". Everything else needs per-bin reporting or size-matching.

### Size-matching — the fix
Resample the training set so its size-bin proportions match the test set.
Then re-measure.

- A gap that **survives** matching is **geographic** — a real transfer failure.
- A gap that **disappears** was **size** all along.

Matching is done **without replacement** deliberately: sampling with
replacement duplicates sites, and duplicated sites inflate apparent sample
size and corrupt the bootstrap intervals. The price is a smaller matched set
(e.g. 7,190 → 3,619 for the Saudi fold). That is the honest trade.

### Leave-one-region-out (LORO)
Three folds instead of one fixed holdout. Two reasons:

1. **Three transfer tests, not one.** A failure appearing in *every* fold
   regardless of which country is held out is about **size**, not geography.
2. **Coverage.** The Saudi fold has 4 very-large sites (unusable). The Iran
   and Iraq folds have 327 and 440. Without LORO that size range would be
   untestable.

---

## 6. INTERMITTENCY — the core scientific idea

**The insight:** flares *flicker*. They are lit, then unlit, then lit again,
so across satellite passes they appear and disappear. Sun-heated desert does
not flicker — it warms with the sun and cools at night, persistently and
seasonally. **That difference in time is what separates a real flare from hot
ground**, and it is why this project fuses a daytime sensor with a nighttime
one instead of just thresholding brightness.

`detection_freq` is the measured version: the fraction of clear satellite
observations in which the flare was detected.

**Measured median across the catalogue: 0.077.** The median catalogued site is
detected in **under 8%** of clear observations. Flares are strongly
intermittent — which supports the method, and confirms that the
continuous-flaring assumption in §2 is generous.

### ⚠️ Units gotcha (found the hard way)
`detection_freq` is **percent (0–100)** in the 2012–2016 and 2018–2021
workbooks, and a **fraction (0–1)** in 2017 and 2022–2024. Mixed together the
column is meaningless. `src/data_eog.py` normalises everything to a fraction
at parse time. Before that fix, the median read as 0.994 — a wrong number that
briefly changed a scientific conclusion.

---

## 7. CROSS-YEAR SITE CLUSTERING

**The problem.** EOG re-issues site IDs every year, and the coordinates jitter
between years because they come from clustered detections, not surveyed
positions. So the same physical flare appears under a different ID and a
slightly different lat/lon each year.

**Why it is not just bookkeeping.** Without stable IDs, a by-site holdout
*leaks*: the same flare sits in train under its 2019 ID and in test under its
2023 ID. That silently breaks RULE 1. (This exact bug was live in this repo
and was fixed in commit `1b8c2ce`.)

**The method.** Complete-linkage agglomerative clustering, cut at **750 m**.

**Why not DBSCAN?** DBSCAN *chains*: with a 750 m radius, a line of flares each
700 m apart merges into one cluster kilometres long — which in a dense Iraqi
oil field would silently fuse distinct flares. **Complete linkage guarantees
no cluster exceeds 750 m in diameter.**

Measured result: **max cluster extent 756 m, median 147 m.** The failure mode
is splitting one flare into two IDs — which costs sample size rather than
inventing false merges. That is the conservative direction.

---

## 8. THE CIRCULARITY CAVEAT — say it before a judge does

**EOG's volumes are themselves produced by a VIIRS calibration.** VIIRS is the
sensor whose blind spots this project targets.

So:
- For sites EOG **contains**, EOG is a **volume reference**.
- For sites EOG **omits**, EOG is **not evidence of anything**. Sentinel-2 is
  the independent channel there.

Say this distinction in every analysis. It is the first thing a competent
judge will probe, and stating it first converts a vulnerability into evidence
of rigour.

---

## 9. THE EIGHT RULES AND WHAT ENFORCES EACH

| # | Rule | Enforced by |
|---|---|---|
| 1 | Never evaluate on training sites/years | `_assert_disjoint()` raises; stable IDs from `build_catalog.py` |
| 2 | Report error by flare size | `assign_size_bins()`; every metric table is per-bin |
| 3 | Bootstrap CI on every headline number | `_bootstrap_ci()`, 10,000 resamples, seed 2718 |
| 4 | Volumes structurally ≥ 0 | `log1p` target / `expm1` inverse — **not** clipping |
| 5 | Simpler model first | LightGBM before any CNN; `cnn.enabled: false` |
| 6 | Baselines before anything new | `baseline.py`, `run_before_anything_else: true` |
| 7 | No magic numbers | every constant lives in `config.yaml` |
| 8 | Log every experiment | `experiment.py` → `experiments/log.csv` |

**On RULE 4** — why `log1p`/`expm1` and not clipping: clipping satisfies the
rule on paper while hiding that the model *wanted* to predict a negative
volume. That is a signal you want to see, not suppress. A structural
transform cannot produce a negative in the first place.

---

## 10. DATA-QUALITY LANDMINES IN THE EOG WORKBOOKS

All found by verification, all currently handled. Any new year must be
re-checked, never assumed.

| Problem | Detail |
|---|---|
| **Country code not stable** | Saudi Arabia is `SAU` in 2012–2023 but `SAUCROP` in 2024. Plus `SAUKWTNZ` for the Neutral Zone. Matching only one code silently dropped 12 years of Saudi data. |
| **Sheet names drift** | `flares_upstream` / `flares upstream` / `flare upstream` |
| **Column names drift** | `Avg_Temp_K` / `Avg. temp., K` / `Avg temp., K` / `Avg. temp` |
| **Missing columns** | 2017 has no `Ellipticity` at all |
| **Shape changes** | 2012–2016 is one *wide* sheet with five years side by side |
| **Unit inconsistency** | `detection_freq` percent vs fraction (§6) |
| **IDs re-issued yearly** | §7 |
| **Country names drift** | "Saudi-Kuwaiti Neutral Zone" vs "Saudi Arabia - Kuwait" |

Columns are matched by **regex**, not exact name — that is what makes all nine
workbooks loadable by one parser.

**A dead source, recorded so nobody re-adds it:** `s3://blackmarble-combustion`
**does not exist** (probed anonymously → `NoSuchBucket`, plus seven variants).

---

## 11. MODULE MAP

| Module | Stage | Does |
|---|---|---|
| `config.py` | 0 | Loads `config.yaml`. Single source of truth. |
| `check_env.py` | 0 | PASS/FAIL/SKIP per dependency + GEE auth |
| `data_eog.py` | 1 | Parses 9 workbooks → `eog_site_year.parquet` |
| `build_catalog.py` | 2 | Cross-year clustering → `catalog.parquet` |
| `data_viirs.py` | — | **stub**, blocked on access path |
| `data_s2.py` | — | **stub**, Sentinel-2 via Earth Engine |
| `features.py` | — | **stub**, intermittency features |
| `splits.py` | 5 | Holdouts + confound diagnostics |
| `baseline.py` | 6 | RULE 6 calibration baseline |
| `model_detect.py` | — | **stub** |
| `model_volume.py` | — | **stub** |
| `evaluate.py` | — | **stub** |
| `experiment.py` | 10 | RULE 8 logging |
| `figures.py` | — | Publication figures |

Run any stage: `python run.py <stage>`

---

## 12. CURRENT HEADLINE NUMBERS

As of 2026-09-01. Regenerate with `python run.py join`.

```
catalogue          11,787 site-years, 2,490 unique sites, 2012–2024
  Saudi Arabia     1,231 sites   4,489 site-years
  Iran               720 sites   4,482 site-years
  Iraq               516 sites   2,661 site-years
  Neutral Zone        23 sites     155 site-years

geography          74.2% of sites inside Ghawar / Basra / Khuzestan,
                   66.6% of total volume — the clustering did not
                   scramble the map

small flares       2024: 320 of 861 sites (37.2%) below 3.1536 Mm³/yr
                   ...but only 0.65% of 2024 volume

intermittency      median detection_freq 0.077

baseline           size-dependent bias, gap 0.466 dex (~2.9×)
                   small/medium overestimated ×1.7
                   very-large underestimated ×4.9
```

### ⚠️ How NOT to use the 37% figure
Those 320 sites **are in the EOG record**. They are not undetected flares, so
this cannot be motivation phrased as "look how many VIIRS misses" — the data
source *is* VIIRS. And they are **0.65% of volume**, so "37% of sites are
small" invites the reply "and under 1% of the gas, so why care?"

**The real motivation** is Seymour's finding that **industry reported 2.2×
more flaring than VIIRS detected** — a gap in flares absent from the catalogue
entirely, which is exactly why Sentinel-2 is needed and EOG cannot answer it.

What the 37% honestly supports: *the sub-threshold size range is real and
populated even inside the official record, so the population beyond it is
unlikely to be empty.* A supporting observation, not the headline.

---

## 13. GLOSSARY

**BCM** — billion cubic metres/year, EOG's unit · **dex** — one power of ten ·
**Bootstrap** — resample the data many times to get an uncertainty range ·
**CI** — confidence interval · **Confounding** — two explanations tangled so
one number can't separate them · **KS** — Kolmogorov–Smirnov, largest gap
between two cumulative distributions · **LORO** — leave-one-region-out ·
**Holdout** — data withheld from training · **Leakage** — test information
reaching the model · **Complete linkage** — clustering that caps cluster
diameter · **SWIR** — short-wave infrared, where flames glow (B11 1610 nm,
B12 2190 nm) · **VIIRS** — the nighttime sensor behind the official record ·
**VNF** — VIIRS Nightfire, the per-detection product carrying radiant heat ·
**EECU-hour** — Earth Engine compute unit; you have 150 · **Radiant heat** —
power radiated by the flare; EOG's actual calibration input, absent from the
public per-site sheets.
