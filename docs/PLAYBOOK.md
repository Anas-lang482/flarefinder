# FlareFinder Playbook

**One ordered guide.** Concepts and impact-multipliers merged into the sequence
you actually work through. Written 2026-09-01.

Each step has the same shape:

> **DO** — the commands · **UNDERSTAND** — the concepts that step produces ·
> **MULTIPLIER** — the upgrade that attaches here · **DONE WHEN** — the check

Steps 1–4 are ✅ already built. **You are at Step 5.**

---
---

# STEP 0 — Read once, applies everywhere

Nothing to run. These four things govern every later step.

## 0.1 Units — get these wrong and every number is wrong

| Unit | Meaning | Conversion |
|---|---|---|
| **BCM** | Billion m³/year. What EOG publishes. | 1 BCM = 1,000 Mm³ |
| **Mm³/yr** | Million m³/year. Our `volume_mcm`. | |
| **m³/h** | m³ per hour. The unit thresholds use. | 1 Mm³/yr ÷ 8,760 h ≈ 114.2 m³/h |

`src/config.py` holds `HOURS_PER_YEAR = 8760.0`. **Never re-derive the
conversion elsewhere** — that is how two modules start disagreeing.

**The Seymour threshold.** VIIRS misses flares below **~360 m³/h**:

```
360 m³/h × 8,760 h = 3,153,600 m³/yr = 3.1536 Mm³/yr
```

That conversion assumes the flare burns **continuously all year**. Most don't
(Step 7). So it is an *upper bound* on the annual equivalent, and counting
sites below it **undercounts** the small-flare population. Say the assumption
out loud every time you use the number.

## 0.2 The circularity caveat — say it before a judge does

**EOG's volumes are themselves produced by a VIIRS calibration.** VIIRS is the
sensor whose blind spots this project targets.

- For sites EOG **contains** → EOG is a **volume reference**.
- For sites EOG **omits** → EOG is **not evidence of anything**. Sentinel-2 is
  the independent channel there.

This is the first thing a competent judge probes. Stating it first turns a
vulnerability into evidence of rigour. It is also the weakness that **Steps 5,
6 and 8's multipliers exist to attack.**

## 0.3 Size bins — why every result is split by flare size

**RULE 2.** A single pooled error number is dominated by the biggest flares —
exactly the ones this project is *not* about.

```
0 ──── 100 ──── 360 ──── 1,000 ──── 5,000 ──── 20,000 ──── ∞
  tiny   sub-VIIRS  small    medium     large     very-large
```

The **360 edge is the Seymour limit**, so `tiny` + `sub-VIIRS` *are* the
sub-threshold regime. A bin needs ≥ **20 test sites** (`min_sites_per_bin`) or
it prints `usable: False` instead of a number — a metric on 4 sites is noise
wearing a number's clothes.

## 0.4 The eight rules and what enforces each

| # | Rule | Enforced by |
|---|---|---|
| 1 | Never evaluate on training sites/years | `_assert_disjoint()` **raises**; stable IDs from `build_catalog.py` |
| 2 | Report error by flare size | `assign_size_bins()`; every table is per-bin |
| 3 | Bootstrap CI on every headline number | `_bootstrap_ci()`, 10,000 resamples, seed 2718 |
| 4 | Volumes structurally ≥ 0 | `log1p`/`expm1` — **not** clipping |
| 5 | Simpler model first | LightGBM before any CNN |
| 6 | Baselines before anything new | `baseline.py` |
| 7 | No magic numbers | everything in `config.yaml` |
| 8 | Log every experiment | `experiment.py` → `experiments/log.csv` |

**On RULE 4:** clipping satisfies the rule on paper while hiding that the model
*wanted* to predict a negative volume. That is a signal you want to see, not
suppress. A structural transform can't produce one at all.

---
---

# STEP 1 — Environment ✅ DONE

## DO
```bash
python run.py check-env
```

## DONE WHEN
`PASS 43 · FAIL 0`. A `SKIP` on rasterio is correct — it's deliberately
unused (Earth Engine returns tables, not rasters).

**Gotcha:** your GEE project id currently comes from the `EE_PROJECT` env var,
which vanishes in a new terminal. `config.yaml` has the same value as backup.

---
---

# STEP 2 — Build the catalogue ✅ DONE

## DO
```bash
python run.py download
```
```bash
python run.py join
```

## UNDERSTAND

### Cross-year site clustering — why it isn't just bookkeeping
EOG **re-issues site IDs every year**, and coordinates jitter between years
because they come from clustered detections, not surveyed positions.

Without stable IDs a by-site holdout **leaks**: the same flare sits in train
under its 2019 ID and in test under its 2023 ID. That silently breaks RULE 1.
*(This exact bug was live in this repo — fixed in commit `1b8c2ce`.)*

**Method:** complete-linkage clustering cut at **750 m**.

**Why not DBSCAN?** DBSCAN *chains* — with a 750 m radius a line of flares
each 700 m apart merges into one cluster kilometres long, which in a dense
Iraqi field would silently fuse distinct flares. Complete linkage
**guarantees** no cluster exceeds 750 m across.

Measured: **max extent 756 m, median 147 m.** The failure mode is splitting one
flare into two IDs — costs sample size, never invents false merges. The
conservative direction.

### The EOG data landmines (all handled; re-check any new year)

| Problem | Detail |
|---|---|
| **Country code unstable** | Saudi is `SAU` in 2012–2023 but `SAUCROP` in 2024, plus `SAUKWTNZ`. Matching one code silently dropped 12 years of data. |
| Sheet names drift | `flares_upstream` / `flares upstream` / `flare upstream` |
| Column names drift | `Avg_Temp_K` / `Avg. temp., K` / `Avg temp., K` / `Avg. temp` |
| Missing columns | 2017 has no `Ellipticity` |
| Shape changes | 2012–2016 is one *wide* sheet, five years side by side |
| Units inconsistent | `detection_freq` percent vs fraction (Step 7) |
| IDs re-issued yearly | see above |

Columns are matched by **regex**, not exact name. That is what makes all nine
workbooks load with one parser.

**A dead source, recorded so nobody re-adds it:**
`s3://blackmarble-combustion` **does not exist** (probed → `NoSuchBucket`,
plus seven variants).

## DONE WHEN
11,787 site-years · 2,490 sites · 2012–2024, and **74.2% of sites fall inside
Ghawar / Basra / Khuzestan** (66.6% of volume). That geographic concentration
is your proof the clustering didn't scramble the map.

## 🔼 MULTIPLIER — do these now, they're cheap

### M7 · Report the data defects to EOG ★★ — 1 hour
Email the EOG / Payne Institute team (Elvidge, Zhizhin) with the `SAUCROP`
change, the `detection_freq` unit inconsistency, and the annual ID re-issue.

Why it multiplies: each is a **real, reproducible defect in a widely used
public dataset** that silently corrupts other people's analyses. A documented
exchange with the data's authors is a strong ISEF credibility signal — and
they may grant VNF access, which is Step 5.

Start now; their response time isn't yours to control.

### M9a · Store the input hashes ★ — 30 minutes
The SHA-256 of each EOG workbook was printed at download but **isn't stored**.
If EOG silently republishes a file, your results shift with no warning. Save
them to the repo now; the rest of M9 waits for Step 13.

---
---

# STEP 3 — Splits and confounding ✅ DONE

## DO
```bash
python run.py splits
```

## UNDERSTAND

### Why holdouts (RULE 1)
Test on data the model never saw, or the score is a memory test.
`_assert_disjoint()` **raises an exception** rather than warning — a silent
leak is worse than a crash.

### The KS statistic — "are train and test even comparable?"
**Kolmogorov–Smirnov:** take the distribution of log volume in train and in
test, measure the **largest vertical gap between their cumulative curves**.
Runs 0 → 1. **0** = identical, **1** = no overlap. **Threshold: 0.150.**

**Why it matters.** Hold out Saudi Arabia and score badly — two explanations:
the model doesn't travel between countries, *or* the model is bad at small
flares and Saudi flares are small. A pooled number can't separate them. KS
measures how tangled they are.

| Split | test n | KS raw | KS matched | Verdict |
|---|---|---|---|---|
| **by-site** | 2,355 | **0.045** | 0.016 | **CLEAN** |
| by-year | 1,647 | 0.209 | 0.049 | confounded |
| region −IRN | 4,487 | 0.195 | 0.020 | confounded |
| region −IRQ | 2,656 | 0.264 | 0.035 | confounded |
| region −SAU | 4,644 | **0.362** | 0.044 | confounded, very-large unusable |

### Which split carries which claim — do not mix these up
- **by-site is the only clean split.** It is the only one that can carry a bare
  "MAE = X ± Y". **Headline numbers come from here.**
- **by-year** carries the forward-monitoring claim, but at KS 0.209 it must be
  reported per size bin or size-matched. **Never quote a pooled by-year number.**
- **region folds** carry the transfer claim *only*, always with matching.
- **Never let one number do both jobs.**

### Size-matching — the fix
Resample training data so its size-bin proportions match the test set.

- A gap that **survives** matching is **geographic** — a real transfer failure.
- A gap that **disappears** was **size** all along.

Done **without replacement** deliberately: with replacement duplicates sites,
and duplicates inflate apparent sample size and corrupt the bootstrap
intervals. Price: a smaller matched set (7,190 → 3,619 on the Saudi fold).
That is the honest trade.

### Leave-one-region-out (LORO)
Three folds, not one fixed holdout:
1. **Three transfer tests.** A failure appearing in *every* fold regardless of
   which country is held out is about **size**, not geography.
2. **Coverage.** The Saudi fold has 4 very-large sites (unusable); the Iran and
   Iraq folds have 327 and 440.

## DONE WHEN
All five splits report, no `RULE 1 VIOLATION` raised.

**Standing limitation for the paper:** no claim about very-large flares
(>20,000 m³/h) in Saudi Arabia is supportable — 4 sites, below the floor.

---
---

# STEP 4 — The baseline ✅ DONE (RULE 6)

## DO
```bash
python run.py baseline
```

## UNDERSTAND — how to read the output table

| size_bin | n | median_log_bias | ci_lo | ci_hi | bias_factor | excludes_zero |
|---|---|---|---|---|---|---|
| tiny | 292 | +0.0876 | 0.0564 | 0.1514 | 1.223 | True |
| sub-VIIRS | 355 | −0.0391 | −0.0790 | 0.0258 | 0.914 | **False** |
| small | 244 | +0.2251 | 0.1304 | 0.2647 | 1.679 | True |
| medium | 347 | +0.2344 | 0.1927 | 0.2748 | 1.715 | True |
| large | 275 | −0.1982 | −0.2321 | −0.1580 | 0.634 | True |
| very-large | 139 | −0.6862 | −0.7240 | −0.6486 | 0.206 | True |

### `median_log_bias` — "how many powers of ten wrong"
`log₁₀(predicted) − log₁₀(actual)`, median across the bin.
**Positive = OVERESTIMATES. Negative = UNDERESTIMATES. Zero = unbiased.**
The unit is a **dex** (decimal exponent): +1 dex = 10× too high, +0.3 ≈ 2×.

**Why log space?** An error of 100 m³/h is catastrophic on a 50 m³/h flare and
irrelevant on a 50,000 m³/h one. A linear residual would be dominated by the
largest flares and tell you nothing about small ones. Log space asks "off by
what *factor*" — the same question at every size.

### `bias_factor` — the same number, readable
`10^(median_log_bias)`. Use the **factor in prose**, the **log bias in tables**.
`1.715` → 1.7× too much. `0.206` → 0.206× the truth, i.e. **~4.9× too little**.

### `ci_lo` / `ci_hi` — the bootstrap 95% CI (RULE 3)
> Resample the test sites **with replacement** 10,000 times, recompute the
> median each time, take the 2.5th and 97.5th percentiles.

It answers: *if I'd drawn a different sample of flares, how much would this
move?* Bootstrap needs no assumption the data is bell-shaped — which matters,
because flare volumes are anything but. Seed 2718, so it's identical on rerun.

### `excludes_zero` — the honesty column
`True` = the interval doesn't contain zero, so the bias is distinguishable
from "none at all."

**`sub-VIIRS` is False** (−0.079 to +0.026). So: *we cannot demonstrate bias in
the sub-VIIRS bin.* **Do not claim one.** That single `False` is worth more
than the five `True`s — it proves the analysis can return "no."

### The headline
```
small bins  (tiny + sub-VIIRS)   +0.024  (×1.06)
large bins  (large + very-large) −0.442  (×0.36)
size-dependent gap               +0.466 dex   ≈ ×2.9
```
Direction matches Elvidge et al. 2024 (small over, large under) — reached
independently on Gulf data.

### ⚠️ THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS
This is **not** a reproduction of EOG's calibration:

- `corr(log temperature, log volume) = +0.025` — carries nothing
- `corr(log ellipticity, log volume) = −0.037` — nothing
- `detection_freq` **alone** → R² 0.831; the other two add **+0.007**
- fitted temperature exponent **−2.81**; Stefan-Boltzmann says ~**+4**. That
  negative value is a **suppression artefact, not physics**

What got fitted is a **detection-frequency regression**. EOG's real formula
multiplies **radiant heat** by a fitted slope (`0.029353`, visible in the
workbook filenames), and neither radiant heat nor source area is in the
per-site sheets.

> **Sayable:** "A calibration built from the observables in the public EOG
> catalogue has strongly size-dependent error."
> **Not sayable:** "We reproduced the standard VIIRS calibration and showed it
> is biased."

**Step 5 is what removes this caveat.**

## 🔼 MULTIPLIER

### M8 · Temporal trend vs Zero Routine Flaring 2030 ★ — 2–3 hours, do now
You already hold 13 years. Is Gulf flaring falling, flat, or rising, against
the World Bank's 2030 commitment? No new pipeline — group `catalog.parquet` by
year and country, bootstrap the trend, plot it.

Gives Step 13 a **second independent finding** and frames the project against
a real international commitment, not just a sensor limitation.

**Watch out:** catalogue coverage grows across years (2019 has 7,209 rows
globally, 2024 has 10,690). A rising *site count* may be better detection, not
more flaring. Compare **volumes**, normalise by coverage, and state which
you're measuring.

---
---

# STEP 5 — VNF per-detection records ⬅ **YOU ARE HERE**

## 🔼 THIS STEP *IS* MULTIPLIER M1 ★★★ — 1–2 days

**What:** VIIRS Nightfire per-overpass records — radiant heat, source
temperature, source area, timestamp. **Per detection, not per year.**

**Why it is the single biggest upgrade available:**

1. **Makes Step 4 real.** With radiant heat you can apply EOG's actual
   published slope (0.029353) and genuinely reproduce their calibration. Today
   the baseline carries a caveat that removes most of its force.
2. **Turns intermittency from annual into per-pass.** `detection_freq` is one
   number per site-year. VNF gives the **actual time series** — which *is* the
   flicker signature your false-positive defence depends on.
3. Unlocks sub-annual analysis: seasonality, shutdowns, sites appearing
   mid-year.

**Blocking issue:** `s3://blackmarble-combustion` does not exist. Two live
candidates:
- **EOG VNF files** at `eogdata.mines.edu` — your account already works. Try
  this first.
- **NASA Black Marble VNP46** via LAADS — needs the Earthdata token
  (Profile → Generate Token; put it in `.env`, which is gitignored).

**Before writing a line of `data_viirs.py`, confirm the source returns real
bytes.** That rule exists because the last documented source didn't.

## DONE WHEN
`data/raw/viirs/` holds real per-detection records, and you can plot one
site's radiant heat across a year.

---
---

# STEP 6 — Sentinel-2 extraction (Phase 2)

## DO
Build `src/data_s2.py`: server-side `reduceRegion` over site buffers,
returning **tables, not rasters**.

**Quota discipline — non-negotiable.** You have **150 EECU-hours**
(Community tier, no billing). Every extraction caches to
`data/processed/` as parquet and is **never re-pulled if the file exists**.
Debug against the cached table, never against GEE. Re-running extraction loops
while debugging downstream code is how the budget disappears.

Bands: **B11 (1610 nm)** and **B12 (2190 nm)** — flames glow in SWIR. **B8A**
is kept as a non-SWIR control.

## 🔼 MULTIPLIER

### M2 · Independent reported volumes ★★★ — 1–2 days, no code
Hunt for flaring volumes that **don't come from satellites**: World Bank
GGFR/GFMR country tables, Saudi Aramco sustainability reports, Iraq Ministry
of Oil / Basra Gas Company, NIOC, and the reported figures Seymour et al. 2025
used.

**This is what breaks the circularity of 0.2.** Right now you can only say "our
model agrees with EOG." With an independent reference you can say **"the
satellite record under-reports relative to what operators themselves report by
X%, and our system recovers Y% of that gap."** That is the sentence Seymour's
2.2× result licenses you to test.

Country-level totals are enough: aggregate your per-site estimates nationally
and compare. Real external validation, no site metering required.

**Honest limit:** reported figures are self-reported and carry their own
biases. Say so. Two imperfect references beat one circular one.

Do it **during** Step 6 while GEE runs. **Must exist before Step 9.**

## DONE WHEN
Every core site has cached SWIR values across the fusion years, and the run
cost well under a third of your EECU budget.

---
---

# STEP 7 — Features: the intermittency signal

## UNDERSTAND — the core scientific idea

**Flares flicker.** Lit, unlit, lit again — so across passes they appear and
disappear. **Sun-heated desert does not flicker**: it warms with the sun and
cools at night, persistently and seasonally.

**That difference in time is what separates a real flare from hot ground.** It
is why this project fuses a daytime sensor with a nighttime one instead of
just thresholding brightness. This is your contribution's engine.

**Measured median `detection_freq`: 0.077.** The median catalogued site is
detected in **under 8%** of clear observations. Flares are strongly
intermittent — which supports the method, and confirms the continuous-flaring
assumption in 0.1 is generous.

### ⚠️ Units gotcha (found the hard way)
`detection_freq` is **percent (0–100)** in 2012–2016 and 2018–2021, but a
**fraction (0–1)** in 2017 and 2022–2024. Mixed, the column is meaningless.
`data_eog.py` normalises to a fraction at parse time. Before that fix the
median read as 0.994 — a wrong number that briefly changed a conclusion.

## DONE WHEN
Features exist for every site-year with **no EOG-derived quantity among the
detection features** — that would be leakage straight into the thing you're
trying to prove.

---
---

# STEP 8 — Detection model (Phase 4)

## DO
LightGBM first (**RULE 5**). A CNN needs explicit justification against it.
Train on the training split only.

## 🔼 MULTIPLIER

### M3 · Negative control / false-positive audit ★★★ — ~1 day, NOT OPTIONAL
Run the finished detector over places with **no known flares**: empty desert,
quarries, steel plants, cement kilns, urban heat, solar farms. Measure the
false-positive rate. **Report it as a headline number.**

Your central claim is "we detect onshore flares VIIRS misses." The obvious
attack is *"how do you know those aren't hot desert or an industrial
furnace?"* The Phase 6 chip review is a spot check; a systematic negative
control is a **measurement**. It converts your weakest point into a quantified
strength — and it's the most likely reason a judge disbelieves your final map.

**The detection model is not finished until this number exists.**

## DONE WHEN
Precision/recall **per size bin** with bootstrap CIs, on the **by-site** split,
plus a stated false-positive rate.

---
---

# STEP 9 — Volume model (Phase 5)

## DO
LightGBM quantile regression at 0.05 / 0.50 / 0.95.
**RULE 4:** `log1p` target, `expm1` inverse — structurally cannot go negative.

## 🔼 MULTIPLIER

### M6 · Calibrated uncertainty, not just intervals ★★ — a few hours
Check the intervals are **actually calibrated**: a 95% interval should contain
the truth 95% of the time. Report measured coverage **per size bin**.
`config.yaml` already lists `coverage_of_prediction_interval`.

Almost every student project reports accuracy. Very few report whether their
stated uncertainty is *honest*. "Our 95% intervals achieve 94.2% coverage on
held-out sites" signals real statistical maturity — and if it comes out at
60%, you've found something important before a judge does.

## DONE WHEN
Error **and** interval coverage per size bin, beaten against the Step 4
baseline, all logged.

---
---

# STEP 10 — Hidden-flare estimate (Phase 6)

## DO
Apply the detector where EOG has nothing. Manually review image chips for
every claimed detection before publishing a count.

## 🔼 MULTIPLIER

### M4 · CO₂e and unburnt-methane conversion ★★ — half a day
Convert detected volumes to CO₂-equivalent and unburnt methane, using
published combustion efficiency (flares ~92–98% efficient; the unburnt
remainder is largely methane, ~28× CO₂ over 100 years).

**This fixes your weakest framing.** "We found 400 uncatalogued flares" is a
satellite result. **"...representing X kt of unaccounted CO₂e, of which Y kt is
unburnt methane"** is a climate result — and methane is where the policy
urgency sits, because a small unburnt volume carries disproportionate warming.

Carry your volume uncertainty **through** the conversion. Report a range, and
state the efficiency assumption as an assumption.

## DONE WHEN
A count **and** a CO₂e figure, both with intervals, and every detection
visually reviewed.

---
---

# STEP 11 — Monitoring + methane (Phase 7)

Automated re-run, plus Sentinel-5P methane cross-check
(`COPERNICUS/S5P/OFFL/L3_CH4`, currently `enabled: false`). An honest null
here is a result — say so plainly if the methane signal isn't there.

## 🔼 MULTIPLIER — optional

### M5 · Landsat 8/9 as a third sensor ★★ — ~1 day
Landsat SWIR (30 m) alongside Sentinel-2. More overpasses → better-resolved
flicker, and a different overpass time gives a partly independent look.
Agreement between two independent daytime sensors is much harder to dismiss.

Only if Steps 1–10 are solid. A strengthener, not a requirement. Watch the
EECU budget.

---
---

# STEP 12 — Application prototype (Phase 8)

Streamlit + folium map of flares invisible to the official record. The wow,
not the science. If time is short, this is the first thing to cut.

---
---

# STEP 13 — Results and paper (Phase 9)

## 🔼 MULTIPLIER

### M9 · Reproducibility package ★ — half a day
`pip freeze > requirements.lock.txt`, the input hashes from M9a, and publish
code + derived catalogue with a DOI (Zenodo, free).

`experiments/log.csv` already ties every number to a git commit and config
hash — this finishes the chain. ISEF-level judges ask whether work reproduces;
a hash-verified, version-pinned package answers in one sentence.

## ⚠️ How NOT to write the motivation
**2024: 320 of 861 sites (37.2%) fall below 3.1536 Mm³/yr.**

Those sites **are in the EOG record**. They are not undetected flares — so this
can't be phrased as "look how many VIIRS misses," because the data source *is*
VIIRS. And they are **0.65% of 2024 volume**, so "37% of sites are small"
invites "and under 1% of the gas, so why care?"

**The real motivation** is Seymour's finding that **industry reported 2.2×
more flaring than VIIRS detected** — a gap in flares absent from the catalogue
entirely, which is exactly why Sentinel-2 is needed and EOG cannot answer it.

What 37% *honestly* supports: *the sub-threshold range is real and populated
even inside the official record, so the population beyond it is unlikely to be
empty.* A supporting observation, not the headline.

---
---

# PRIORITY SUMMARY

```
NOW, in parallel ......... M7  email EOG about their data defects   (Step 2)
                           M8  temporal trend, data already on disk (Step 4)
                           M9a store the input hashes               (Step 2)

STEP 5  ⬅ HERE .......... M1  VNF per-detection records        ★★★
STEP 6 ................... M2  independent reported volumes     ★★★
STEP 8 ................... M3  negative control audit           ★★★
STEP 9 ................... M6  interval coverage
STEP 10 .................. M4  CO2e + methane conversion        ★★
STEP 11 .................. M5  Landsat third sensor (optional)
STEP 13 .................. M9  reproducibility package
```

**If you only do three: M1, M2, M3.** M1 makes the baseline real. M2 breaks the
circularity. M3 defends the central claim.

**Scope discipline.** A bulletproof Steps 1–10 beats a shaky 1–12. A multiplier
that destabilises the core is not a multiplier. Drop any still unfinished two
weeks before the deadline.

---
---

# APPENDIX A — Module map

| Module | Does |
|---|---|
| `config.py` | Loads `config.yaml`. Single source of truth. |
| `check_env.py` | PASS/FAIL/SKIP per dependency + GEE auth |
| `data_eog.py` | Parses 9 workbooks → `eog_site_year.parquet` |
| `build_catalog.py` | Cross-year clustering → `catalog.parquet` |
| `splits.py` | Holdouts + confound diagnostics |
| `baseline.py` | RULE 6 calibration baseline |
| `experiment.py` | RULE 8 logging |
| `figures.py` | Publication figures |
| `data_viirs.py` | **stub** — Step 5 |
| `data_s2.py` | **stub** — Step 6 |
| `features.py` | **stub** — Step 7 |
| `model_detect.py` | **stub** — Step 8 |
| `model_volume.py` | **stub** — Step 9 |
| `evaluate.py` | **stub** — Steps 8–9 |

Run any stage: `python run.py <stage>`

---

# APPENDIX B — Current headline numbers (2026-09-01)

```
catalogue        11,787 site-years · 2,490 sites · 2012–2024
  Saudi Arabia   1,231 sites   4,489 site-years
  Iran             720 sites   4,482 site-years
  Iraq             516 sites   2,661 site-years
  Neutral Zone      23 sites     155 site-years

geography        74.2% of sites in Ghawar / Basra / Khuzestan
                 66.6% of total volume

small flares     2024: 320 of 861 sites (37.2%) below 3.1536 Mm³/yr
                 ...but only 0.65% of 2024 volume  ← read Step 13

intermittency    median detection_freq 0.077

baseline         size-dependent bias, gap 0.466 dex (≈2.9×)
                 small/medium overestimated ×1.7
                 very-large underestimated ×4.9
                 sub-VIIRS bin: CI spans zero — no bias demonstrable
```

---

# APPENDIX C — Glossary

**BCM** — billion m³/year, EOG's unit · **dex** — one power of ten ·
**Bootstrap** — resample many times to get an uncertainty range · **CI** —
confidence interval · **Confounding** — two explanations tangled so one number
can't separate them · **KS** — Kolmogorov–Smirnov, largest gap between two
cumulative distributions · **LORO** — leave-one-region-out · **Holdout** — data
withheld from training · **Leakage** — test information reaching the model ·
**Complete linkage** — clustering that caps cluster diameter · **SWIR** —
short-wave infrared, where flames glow (B11 1610 nm, B12 2190 nm) · **VIIRS** —
the nighttime sensor behind the official record · **VNF** — VIIRS Nightfire,
the per-detection product carrying radiant heat · **EECU-hour** — Earth Engine
compute unit; you have 150 · **Radiant heat** — power radiated by the flare;
EOG's actual calibration input, absent from the public per-site sheets.
