# FlareFinder — Complete Playbook
## From zero to a working AI model + application prototype
### "Seeing the Invisible Waste: Detecting Undetected Gas Flares Across the Gulf"

**The single guide.** It merges the original build guide (goals, manual steps,
ready-to-paste Claude Code prompts, success checks, pitfalls), the concepts
needed to read every number the pipeline prints, the impact multipliers placed
at the step where each belongs, and **every correction found by actually
running it**.

Last updated 2026-09-01. Steps 0–4 are ✅ built. **You are at Step 5.**

---

## HOW TO USE THIS GUIDE

1. `CLAUDE.md` in the project root holds the project context — Claude Code
   reads it automatically, so every prompt is already informed by it. Keep it
   updated as decisions change.
2. Work through steps in order. Paste each step's prompt when you reach it.
3. Run the success check. **Don't move on until it passes.**
4. Commit after every working step.
5. Where a ⚠️ **CORRECTION** box appears, reality differed from the original
   plan. Those are the expensive lessons — read them before running the step.

Each step has the same shape:

> **GOAL** · **YOU DO MANUALLY** · **PROMPT** · **UNDERSTAND** (concepts that
> step produces) · 🔼 **MULTIPLIER** (the upgrade that attaches here) ·
> **SUCCESS CHECK**

Commands are PowerShell-safe (no `&&`).

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

That assumes the flare burns **continuously all year**. Most don't (Step 7).
So it is an *upper bound* on the annual equivalent, and counting sites below
it **undercounts** the small-flare population. State the assumption every time.

## 0.2 The circularity caveat — say it before a judge does

**EOG's volumes are themselves produced by a VIIRS calibration.** VIIRS is the
sensor whose blind spots this project targets.

- For sites EOG **contains** → EOG is a **volume reference**.
- For sites EOG **omits** → EOG is **not evidence of anything**. Sentinel-2 is
  the independent channel there.

First thing a competent judge probes. Stating it first turns a vulnerability
into evidence of rigour. It is also the weakness that **the three ★★★
multipliers exist to attack.**

## 0.3 Size bins — why every result is split by flare size

**RULE 2.** A single pooled error number is dominated by the biggest flares —
exactly the ones this project is *not* about.

```
0 ──── 100 ──── 360 ──── 1,000 ──── 5,000 ──── 20,000 ──── ∞
  tiny   sub-VIIRS  small    medium     large     very-large
```

The **360 edge is the Seymour limit**, so `tiny` + `sub-VIIRS` *are* the
sub-threshold regime. A bin needs ≥ **20 test sites** (`min_sites_per_bin`) or
it prints `usable: False` — a metric on 4 sites is noise wearing a number's
clothes.

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
*wanted* to predict a negative volume. That's a signal you want to see, not
suppress. A structural transform can't produce one at all.

---
---

# STEP 1 — Environment & accounts ✅ DONE

**GOAL:** working environment, all accounts requested, repo with structure.

**YOU DID MANUALLY:** registered Google Earth Engine (noncommercial, Community
tier), EOG, NASA Earthdata; installed Python 3.12, git, VS Code, Claude Code;
ran `earthengine authenticate`.

**PROMPT USED**
```
Set up the project skeleton exactly as described in CLAUDE.md: create the
repo layout, a config.yaml with placeholders for region bounds (Saudi
Arabia, Iraq, Iran bounding boxes), years, random seed 42, and a
requirements.txt with the stack listed in CLAUDE.md. Create a Makefile or
simple run.py with commands for each future pipeline stage (download, join,
features, baseline, train, evaluate, app). Initialize git with a sensible
.gitignore for data/ and models/. Then write src/check_env.py that verifies
every library imports and that Earth Engine authenticates, printing
PASS/FAIL per item. Run it and fix whatever fails.
```

**DO**
```bash
python run.py check-env
```

⚠️ **CORRECTIONS FOUND**
- **`run.py`, not a Makefile.** Windows has no `make`. A Python entry point
  runs identically in PowerShell, Git Bash and CMD.
- **Years 2019–2025 → 2019–2024.** EOG has not published 2025. A half-filled
  final year would bias every size-binned metric.
- **rasterio excluded.** Windows Application Control blocked its native DLL,
  and nothing uses it — Earth Engine returns tables, not rasters. Reported as
  an optional `SKIP`, never a `FAIL`, so the FAIL column stays meaningful.
- **`openpyxl` was missing** from the original stack list. The ground truth
  ships as `.xlsx`; without it Step 2 cannot read anything.
- **GEE Community tier**, not Contributor: 150 EECU-hours, **no billing
  account required**. Never attach a card to this project.

**SUCCESS CHECK:** `PASS 43 · FAIL 0`.

**Gotcha:** your GEE project id currently comes from the `EE_PROJECT` env var,
which vanishes in a new terminal. `config.yaml` holds the same value as backup.

---
---

# STEP 2 — Ground truth: the EOG flare catalogue ✅ DONE

**GOAL:** a clean table of every known flare site in Saudi Arabia, Iraq and
Iran with coordinates and annual volumes — the study population and the
regression targets.

**YOU DID MANUALLY:** nothing, as it turned out — see the correction below.

**PROMPT USED**
```
In src/build_catalog.py: load every EOG annual flare-site xlsx in
data/raw/eog/. These files have per-site rows with lat, lon, country, and
estimated flared volume (bcm or million m3 — check units per file and
normalize to million m3/year, documenting the conversion). Filter to Saudi
Arabia, Iraq, Iran. Deduplicate sites across years by clustering coordinates
within 750 m (flare coordinates jitter between years) and assign a stable
site_id. Output data/processed/catalog.parquet with: site_id, lat, lon,
country, year, volume_mcm, n_years_observed. Print a summary: sites per
country, volume distribution (log-histogram), and how many sites fall below
the Seymour small-flare threshold. Then build figures/fig01_flare_map.html:
a folium map of all sites, marker size by volume, colored by country. Also
export a static PNG version for the paper.
```

**DO**
```bash
python run.py download
```
```bash
python run.py join
```
```bash
python run.py figures
```

## UNDERSTAND

### Cross-year site clustering — why it isn't just bookkeeping
EOG **re-issues site IDs every year**, and coordinates jitter between years
because they come from clustered detections, not surveyed positions.

Without stable IDs a by-site holdout **leaks**: the same flare sits in train
under its 2019 ID and in test under its 2023 ID. That silently breaks RULE 1.
*(This exact bug was live in this repo — fixed in `1b8c2ce`.)*

**Method:** complete-linkage clustering cut at **750 m**.

**Why not DBSCAN?** DBSCAN *chains* — with a 750 m radius a line of flares each
700 m apart merges into one cluster kilometres long, which in a dense Iraqi
field would silently fuse distinct flares. Complete linkage **guarantees** no
cluster exceeds 750 m across.

Measured: **max extent 756 m, median 147 m.** The failure mode is splitting one
flare into two IDs — costs sample size, never invents false merges. The
conservative direction.

### Units, verified not assumed
All nine workbooks report **BCM**; there is no million-m³ variant despite the
prompt allowing for one. So `volume_mcm = BCM × 1000`. Sanity check: Iraq 2024
sums to 18.16 BCM, matching the published World Bank figure.

⚠️ **CORRECTIONS FOUND — the EOG data landmines**

| Problem | Detail |
|---|---|
| **Country code unstable** | Saudi is `SAU` in 2012–2023 but **`SAUCROP` in 2024 only**, plus `SAUKWTNZ` for the Neutral Zone. Matching one code silently dropped 12 years of Saudi data — 295 site-years instead of 4,489. **Any new year must be re-checked, never assumed.** |
| Sheet names drift | `flares_upstream` / `flares upstream` / `flare upstream` |
| Column names drift | `Avg_Temp_K` / `Avg. temp., K` / `Avg temp., K` / `Avg. temp` |
| Missing columns | 2017 has no `Ellipticity` at all |
| Shape changes | 2012–2016 is one *wide* sheet, five years side by side |
| Units inconsistent | `detection_freq` percent vs fraction — see Step 7 |
| IDs re-issued yearly | see clustering above |
| Country names drift | "Saudi-Kuwaiti Neutral Zone" vs "Saudi Arabia - Kuwait" |

Columns are matched by **regex**, not exact name. That is what makes all nine
workbooks load with one parser.

**No login needed.** The annual xlsx files are public direct links at
`eogdata.mines.edu/global_flare_data/`; the EOG account is not required for
this step (it may still be for VNF in Step 5).

**SUCCESS CHECK:** `catalog.parquet` exists — 11,787 site-years, 2,490 sites,
2012–2024 — and the map shows sites clustered in the known oil regions.
Verified numerically: **74.2% of sites inside Ghawar / Basra / Khuzestan**,
66.6% of volume. That concentration is your proof the clustering didn't
scramble the map.

## 🔼 MULTIPLIERS — cheap, do them now

### M7 · Report the data defects to EOG ★★ — 1 hour
Email the EOG / Payne Institute team (Elvidge, Zhizhin) with the `SAUCROP`
change, the `detection_freq` unit inconsistency, and the annual ID re-issue.

Each is a **real, reproducible defect in a widely used public dataset** that
silently corrupts other people's analyses. A documented exchange with the
data's authors is a strong ISEF credibility signal — and they may grant VNF
access, which is Step 5. Start now; their response time isn't yours to control.

### M9a · Store the input hashes ★ — 30 minutes
The SHA-256 of each workbook was printed at download but **isn't stored**. If
EOG silently republishes a file, your results shift with no warning.

---
---

# STEP 3 — Splits and confounding ✅ DONE

**GOAL:** holdouts that make RULE 1 enforceable, and a measurement of how far
each split can be trusted.

*This step did not exist in the original guide. It was added after the region
holdout turned out to be confounded with flare size.*

**DO**
```bash
python run.py splits
```

## UNDERSTAND

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
- **by-site is the only clean split.** The only one that can carry a bare
  "MAE = X ± Y". **Headline numbers come from here.**
- **by-year** carries the forward-monitoring claim, but at KS 0.209 must be
  per-size-bin or size-matched. **Never quote a pooled by-year number.**
- **region folds** carry the transfer claim *only*, always with matching.
- **Never let one number do both jobs.**

### Size-matching — the fix
Resample training data so its size-bin proportions match the test set.
- A gap that **survives** matching is **geographic** — a real transfer failure.
- A gap that **disappears** was **size** all along.

Done **without replacement** deliberately: with replacement duplicates sites,
and duplicates inflate apparent sample size and corrupt the bootstrap
intervals. Price: a smaller matched set (7,190 → 3,619 on the Saudi fold).

### Leave-one-region-out (LORO)
Three folds, not one fixed holdout:
1. **Three transfer tests.** A failure appearing in *every* fold regardless of
   which country is held out is about **size**, not geography.
2. **Coverage.** The Saudi fold has 4 very-large sites (unusable); the Iran and
   Iraq folds have 327 and 440.

**SUCCESS CHECK:** all five splits report, no `RULE 1 VIOLATION` raised.

**Standing limitation for the paper:** no claim about very-large flares
(>20,000 m³/h) in Saudi Arabia is supportable — 4 sites, below the floor.

---
---

# STEP 4 — Baselines: reproduce the failure you claim to fix ✅ DONE

**GOAL:** implement the standard VIIRS volume calibration and measure where it
fails, by flare size. This is the scientific heart of your motivation.

**PROMPT USED**
```
In src/baseline.py: implement the standard VIIRS Nightfire volume calibration
(Elvidge-style: flared volume as a function of radiant heat, using the
published functional form and coefficients — cite them in comments; if the
exact published coefficients are ambiguous, fit the SAME functional form on
our training years and say so explicitly). Apply it to our VNF signals to
produce per-site annual volume estimates. Compare against EOG catalog volumes
on held-out data. Produce a predicted-vs-EOG scatter, an error-vs-size figure
with bootstrap 95% CIs, and a detection-side invisibility table. Write results
to experiments/log.csv. Report honestly: our claim is that error is worst for
small flares — check whether OUR data shows that. If it does not, flag it
loudly.
```

**DO**
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

*If I'd drawn a different sample of flares, how much would this move?*
Bootstrap needs no assumption the data is bell-shaped — which matters, because
flare volumes are anything but. Seed 2718, identical on rerun.

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

⚠️ **CORRECTION — THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS**

The original prompt assumed radiant heat would be available. **It is not in the
public per-site sheets.** So this is **not** a reproduction of EOG's
calibration:

- `corr(log temperature, log volume) = +0.025` — carries nothing
- `corr(log ellipticity, log volume) = −0.037` — nothing
- `detection_freq` **alone** → R² 0.831; the other two add **+0.007**
- fitted temperature exponent **−2.81**; Stefan-Boltzmann says ~**+4**. That
  negative value is a **suppression artefact, not physics**

What got fitted is a **detection-frequency regression**. EOG's real formula
multiplies **radiant heat** by a fitted slope (`0.029353`, visible in the
workbook filenames).

> **Sayable:** "A calibration built from the observables in the public EOG
> catalogue has strongly size-dependent error."
> **Not sayable:** "We reproduced the standard VIIRS calibration and showed it
> is biased."

**Step 5 is what removes this caveat.**

**SUCCESS CHECK:** error-vs-size shows degradation, and it does — 0.466 dex,
five of six bins with CIs excluding zero. Whatever it showed, you now have a
measured baseline your model must beat.

## 🔼 MULTIPLIER

### M8 · Temporal trend vs Zero Routine Flaring 2030 ★ — 2–3 hours, do now
You already hold 13 years. Is Gulf flaring falling, flat or rising, against the
World Bank's 2030 commitment? No new pipeline — group `catalog.parquet` by year
and country, bootstrap the trend, plot it.

Gives Step 13 a **second independent finding** and frames the project against a
real international commitment, not just a sensor limitation.

**Watch out:** catalogue coverage grows across years (2019 has 7,209 rows
globally, 2024 has 10,690). A rising *site count* may be better detection, not
more flaring. Compare **volumes**, normalise by coverage, say which you measure.

---
---

# STEP 5 — VNF nighttime records ⬅ **YOU ARE HERE**

**GOAL:** per-overpass VIIRS Nightfire records — radiant heat, source
temperature, source area, timestamp. Per detection, not per year.

## 🔼 THIS STEP *IS* MULTIPLIER M1 ★★★ — 1–2 days

**Why it is the single biggest upgrade available:**

1. **Makes Step 4 real.** With radiant heat you can apply EOG's published slope
   (0.029353) and genuinely reproduce their calibration, removing the caveat
   that currently guts the baseline.
2. **Turns intermittency from annual into per-pass.** `detection_freq` is one
   number per site-year. VNF gives the **actual time series** — which *is* the
   flicker signature your false-positive defence depends on.
3. Unlocks sub-annual analysis: seasonality, shutdowns, sites appearing
   mid-year.

**ORIGINAL PROMPT** (kept for reference — the source in it is wrong)
```
In src/extract_vnf.py: pull the Black Marble / VIIRS nighttime combustion
detections for our region and years. Primary source: the AWS open bucket
(s3://blackmarble-combustion ... accessed with --no-sign-request via boto3
anonymous config). ... For every site/control and month, compute: number of
nighttime detections within 750 m, mean/max radiant heat if available. Output
data/processed/vnf_timeseries.parquet. Sanity check: detection counts at known
large flares should be far higher than at controls — print that comparison.
```

⚠️ **CORRECTION — THE SOURCE IN THAT PROMPT DOES NOT EXIST**
`s3://blackmarble-combustion` returns `NoSuchBucket` on anonymous access, as do
seven plausible variants. Do not build against it.

Two live candidates, in order:
1. **EOG VNF files** at `eogdata.mines.edu` — your account already works.
2. **NASA Black Marble VNP46** via LAADS — needs the Earthdata token
   (Profile → Generate Token; store in `.env`, which is gitignored).

**Rule for this step: confirm the source returns real bytes BEFORE writing a
line of `data_viirs.py`.** That rule exists because the documented source
didn't.

**SUCCESS CHECK:** `data/raw/viirs/` holds real per-detection records, and you
can plot one site's radiant heat across a year. Detection counts at known large
flares must be far higher than at controls — print that table.

---
---

# STEP 6 — Sentinel-2 signals for every site (Phase 2)

**GOAL:** for each site and matched control point, monthly Sentinel-2 SWIR
statistics. This is the dataset everything trains on.

**PROMPT A — controls & sampling design**
```
In src/make_controls.py: for each flare site in catalog.parquet, generate 3
control points 5-15 km away that are NOT within 2 km of any known flare site
(any country, any year). Controls must sample the same desert/land surface, so
keep them onshore (use a land mask via GEE or natural-earth polygons). Output
data/processed/controls.parquet with control_id, lat, lon, paired_site_id.
Print counts and add controls to the folium map in a second layer (gray).
```

**PROMPT B — Sentinel-2 SWIR extraction via GEE**
```
In src/extract_s2.py: using the earthengine-api with batched server-side
reduceRegions (never per-image Python loops — design for ~1500 points x ~6
years), extract for every site and control point a monthly time series from
COPERNICUS/S2_SR_HARMONIZED: mean/max of B11 and B12 in a 100 m buffer, the
normalized SWIR ratio (B12-B11)/(B12+B11), cloud fraction (from SCL band), and
observation count. Handle GEE quota limits with chunking and exponential
backoff retries; cache completed chunks to data/processed/s2_chunks/ so the job
resumes if interrupted. Merge into data/processed/s2_timeseries.parquet keyed
by (point_id, year, month). Validate: for the 20 largest-volume sites, plot B12
max vs time — we expect persistently elevated SWIR at big flares. Save as
figures/fig02_s2_sanity.png. If big flares do NOT show elevated SWIR, STOP and
tell me — that invalidates the approach and we must debug (buffer size, cloud
masking, or coordinate issues) before continuing.
```

**QUOTA DISCIPLINE — non-negotiable.** You have **150 EECU-hours** (Community
tier, no billing). Every extraction caches to parquet and is **never re-pulled
if the file exists**. Debug against the cached table, never against GEE.
Re-running extraction loops while debugging downstream code is how the budget
disappears.

Bands: **B11 (1610 nm)** and **B12 (2190 nm)** — flames glow in SWIR. **B8A**
is kept as a non-SWIR control band.

## 🔼 MULTIPLIER

### M2 · Independent reported volumes ★★★ — 1–2 days, no code
Hunt for flaring volumes that **don't come from satellites**: World Bank
GGFR/GFMR country tables, Saudi Aramco sustainability reports, Iraq Ministry of
Oil / Basra Gas Company, NIOC, and the reported figures Seymour et al. 2025
used.

**This is what breaks the circularity of 0.2.** Right now you can only say "our
model agrees with EOG." With an independent reference you can say **"the
satellite record under-reports relative to what operators themselves report by
X%, and our system recovers Y% of that gap."** That is the sentence Seymour's
2.2× result licenses you to test.

Country-level totals suffice: aggregate your per-site estimates nationally and
compare. Real external validation, no site metering required.

**Honest limit:** reported figures are self-reported with their own biases. Say
so. Two imperfect references beat one circular one.

Do it **during** Step 6 while GEE runs. **Must exist before Step 9.**

**SUCCESS CHECK:** two time-series tables; sanity figures show big flares
glowing in SWIR and lighting up at night, controls dark in both. Run cost well
under a third of the EECU budget.

---
---

# STEP 7 — Features: the intermittency signal

**GOAL:** the fused feature table, including the flicker features that are the
project's engine.

**PROMPT**
```
In src/features.py: build features per point-year from the two time-series
tables. SWIR level stats (mean/max/p95 of B11, B12, ratio). TEMPORAL features
that encode flicker — coefficient of variation of monthly B12 max, fraction of
months with B12 above a locally-learned percentile threshold, lag-1
autocorrelation (hot ground is seasonally smooth and sun-driven; flares switch
on/off), day-night agreement (S2 elevated AND/OR VNF detections), VNF detection
count and radiant heat stats, and site context (distance to nearest known flare
cluster). No leakage: nothing derived from EOG volume enters detection features.
```

## UNDERSTAND — the core scientific idea

**Flares flicker.** Lit, unlit, lit again — so across passes they appear and
disappear. **Sun-heated desert does not flicker**: it warms with the sun and
cools at night, persistently and seasonally.

**That difference in time is what separates a real flare from hot ground.** It
is why this project fuses a daytime sensor with a nighttime one instead of just
thresholding brightness. This is your contribution's engine.

**Measured median `detection_freq`: 0.077.** The median catalogued site is
detected in **under 8%** of clear observations. Flares are strongly
intermittent — which supports the method, and confirms the continuous-flaring
assumption in 0.1 is generous.

⚠️ **CORRECTION — units gotcha, found the hard way**
`detection_freq` is **percent (0–100)** in 2012–2016 and 2018–2021, but a
**fraction (0–1)** in 2017 and 2022–2024. Mixed, the column is meaningless.
`data_eog.py` normalises to a fraction at parse time. Before that fix the median
read as 0.994 — a wrong number that briefly changed a scientific conclusion.

**SUCCESS CHECK:** features exist for every site-year with **no EOG-derived
quantity among the detection features** — that would be leakage straight into
the thing you're trying to prove.

---
---

# STEP 8 — The detection model (Phase 4)

**GOAL:** classify flare vs non-flare from fused features. Headline metric:
recall on SMALL flares vs nighttime-only detection.

**PROMPT**
```
In src/train_detector.py:
MODEL: gradient-boosted trees (LightGBM), binary flare vs control.
SPLITS: GroupKFold by site_id (a site never appears in both train and test) AND
a temporal split. Report both.
METRICS: overall ROC-AUC and PR-AUC, but the HEADLINE is recall at 95%
precision, broken out by EOG volume decile — especially the smallest deciles
and sites with zero VNF detections (the invisible ones). Compare against the
nighttime-only rule (VNF detections > 0) as the competitor. Ablation: with vs
without the temporal flicker features — we claim flicker kills desert false
positives; measure the precision impact. Bootstrap CIs on all headline numbers.
SHAP summary plot to figures/fig05_detector_shap.png; results table to
figures/tab01_detection.csv and experiments/log.csv.
```

**RULE 5:** LightGBM first. A CNN needs explicit justification against it.

## 🔼 MULTIPLIER

### M3 · Negative control / false-positive audit ★★★ — ~1 day, NOT OPTIONAL
Run the finished detector over places with **no known flares**: empty desert,
quarries, steel plants, cement kilns, urban heat, solar farms. Measure the
false-positive rate. **Report it as a headline number.**

Your central claim is "we detect onshore flares VIIRS misses." The obvious
attack is *"how do you know those aren't hot desert or an industrial furnace?"*
The Step 10 chip review is a spot check; a systematic negative control is a
**measurement**. It converts your weakest point into a quantified strength —
and it's the most likely reason a judge disbelieves your final map.

**The detection model is not finished until this number exists.**

**SUCCESS CHECK:** detector beats the nighttime-only rule on small-flare recall
at matched precision, with CIs. The flicker ablation shows a measurable
precision gain — **if not, that's a finding; report it honestly.**

---
---

# STEP 9 — The volume model (Phase 5)

**GOAL:** learned calibration, physics-constrained, uncertainty on every
estimate. Must beat the Step 4 baseline, especially on small flares.

**PROMPT**
```
In src/train_volume.py: regression from fused features to EOG annual volume on
EOG-cataloged sites.
PHYSICS CONSTRAINT: model log(volume) so predictions are structurally positive.
MODELS: LightGBM quantile regression (q10/q50/q90) as primary; optional small
PyTorch MLP as ablation only if LightGBM plateaus.
UNCERTAINTY: calibrate the q10-q90 interval with split-conformal adjustment on
a calibration fold so empirical coverage on held-out data is ~80%; report
achieved coverage overall AND for the smallest size decile separately (coverage
often breaks exactly where it matters — measure it, don't assume it).
SPLITS: same protocol as Step 8.
EVALUATION: relative error vs size deciles with CIs, side by side with the Step
4 baseline on IDENTICAL held-out data — figures/fig06_volume_vs_baseline.png is
the single most important figure of the project. Also report transfer: train on
2 countries, test on the third. Log everything to experiments/log.csv.
```

**RULE 4:** `log1p` target, `expm1` inverse — structurally cannot go negative.

## 🔼 MULTIPLIER

### M6 · Calibrated uncertainty, not just intervals ★★ — a few hours
Check the intervals are **actually calibrated**: a 95% interval should contain
the truth 95% of the time. Report measured coverage **per size bin**.
`config.yaml` already lists `coverage_of_prediction_interval`.

Almost every student project reports accuracy. Very few report whether their
stated uncertainty is *honest*. "Our 95% intervals achieve 94.2% coverage on
held-out sites" signals real statistical maturity — and if it comes out at 60%,
you've found something important before a judge does.

**SUCCESS CHECK:** your error-vs-size curve sits below the baseline's with
non-overlapping CIs at least in the small bins; conformal coverage within a few
points of nominal.

⚠️ **FALLBACK:** if the model does NOT beat the baseline by end of Week 6,
execute the proposal's fallback — the methane track becomes primary, and the
negative result gets reported honestly.

---
---

# STEP 10 — The hidden-flare estimate (Phase 6)

**GOAL:** the number nobody has — how much Gulf flaring is invisible to the
official record.

**PROMPT**
```
In src/hidden_flares.py: run the trained detector over a systematic grid of
candidate locations in the study region's oil zones (buffered union around
known flare clusters plus oil-field polygons if available; document the
definition). Flag high-confidence detections NOT in the EOG catalog (>=1.5 km
from any cataloged site). For each, estimate volume + conformal interval.
Aggregate: total hidden volume per country with propagated uncertainty
(bootstrap over sites). Convert to dollars (gas price as a RANGE, e.g. $2-8
per MMBtu, document m3→MMBtu) and CO2e (state the emission factor source).
Output data/processed/hidden_flares.parquet and figures/fig07_hidden_map.html
(red = hidden, gray = cataloged).
CRITICAL honesty pass: manually review the top 20 hidden detections in
Sentinel-2 imagery (export S2 chips to figures/chips/) — industrial heat
sources like steel plants or gas processing can masquerade as flares. Label
each chip plausible-flare / industrial / unclear, and report the plausibility
rate alongside the headline number.
```

## 🔼 MULTIPLIER

### M4 · CO₂e and unburnt-methane conversion ★★ — half a day
Convert detected volumes to CO₂-equivalent and unburnt methane using published
combustion efficiency (flares ~92–98% efficient; the unburnt remainder is
largely methane, ~28× CO₂ over 100 years).

**This fixes your weakest framing.** "We found 400 uncatalogued flares" is a
satellite result. **"...representing X kt of unaccounted CO₂e, of which Y kt is
unburnt methane"** is a climate result — and methane is where the policy
urgency sits, because a small unburnt volume carries disproportionate warming.

Carry your volume uncertainty **through** the conversion. Report a range, and
state the efficiency assumption as an assumption.

**SUCCESS CHECK:** a headline sentence like *"we detect N probable flares
absent from the official record, totalling X ± Y million m³/year (~$A–B
million, ~C kt CO₂e)"* — with the manual-review plausibility rate attached.

---
---

# STEP 11 — Monitoring pipeline + methane extension (Phase 7)

**GOAL:** turn the models into a system.

**PROMPT — monitoring**
```
In src/monitor.py: build a run_monitor(date_range) function that queries GEE
for new S2 scenes over the study region since the last run (state in
data/processed/monitor_state.json), recomputes features for all tracked +
hidden sites, runs the detector, and emits a change report: NEW detections,
sites gone dark, volume-trend flags. Output both machine-readable JSON and a
human-readable markdown report to reports/. Add a --simulate mode that replays
2024 month by month (this becomes the demo for judges: "here is January… here
a new flare appears in March").
```

**PROMPT — methane extension** *(only if Steps 8–10 are solid)*
```
In src/methane.py: for our detected sites, extract Sentinel-5P L3 CH4 (GEE
COPERNICUS/S5P/OFFL/L3_CH4) monthly means in site-centered windows vs matched
upwind background windows, and test for site-associated enhancement. Be
conservative: S5P pixels are ~7 km, far coarser than flares, so
individual-site attribution is weak — frame results as regional/cluster-level
enhancement, with significance testing and multiple-comparison correction.
Where published labeled plume datasets (Schuit 2023 Zenodo, NASA EMIT) contain
plumes in our region, cross-reference: do any coincide with our hidden
detections? Report as a table with clear caveats. If signals are too weak, say
so — a careful null result here does not harm the core project.
```

## 🔼 MULTIPLIER — optional

### M5 · Landsat 8/9 as a third sensor ★★ — ~1 day
Landsat SWIR (30 m) alongside Sentinel-2. More overpasses → better-resolved
flicker, and a different overpass time gives a partly independent look.
Agreement between two independent daytime sensors is much harder to dismiss.
Only if Steps 1–10 are solid. Watch the EECU budget.

**SUCCESS CHECK:** `python run.py monitor --simulate 2024` produces monthly
change reports; the methane section produces either a cautious positive finding
or an honest null.

---
---

# STEP 12 — The application prototype (Phase 8)

**GOAL:** the public-facing artifact judges can touch.

**PROMPT**
```
In app/: build a Streamlit application "FlareFinder" with:
1. MAP page (main): folium map of the Gulf. Layers: cataloged flares (gray),
   hidden flares we detected (red), controls off by default. Click a site →
   sidebar shows its S2/VNF time series, model volume estimate with the
   conformal interval, detection confidence, and the S2 image chip. Filters:
   country, size class, year, hidden-only toggle.
2. IMPACT page: headline counters computed live from hidden_flares.parquet —
   hidden sites found, hidden volume/year with interval, dollar range, CO2e —
   each with a "how computed" expander stating every assumption.
3. MONITOR page: renders the latest reports/ markdown and a month-slider replay
   of the 2024 simulation.
4. METHODS page: honest one-screen summary — data sources, the EOG circularity
   caveat, validation protocol, and limitations, for a non-expert judge.
Design: clean, fast, works offline from local parquet files (no live GEE calls).
Arabic/English toggle for headline text if straightforward. Add app/README.md
with one-command run instructions. Test end to end with our actual data files.
```

**SUCCESS CHECK:** the app runs locally and a judge can click a red dot and see
*why* the model believes it's a flare. If time is short, this is the first
thing to cut.

---
---

# STEP 13 — Results, figures, paper (Phase 9)

**GOAL:** consolidate everything into the Ebdaa paper + poster material.

**PROMPT**
```
In src/make_paper_assets.py: regenerate all publication figures at 300 DPI with
consistent styling and numbered filenames: fig01 study-region flare map, fig04
baseline error-vs-size, fig05 detector SHAP, fig06 volume model vs baseline
(the headline), fig07 hidden-flare map, plus tab01-tab04: detection metrics,
volume metrics with coverage, country transfer, hidden-flare totals with
assumptions. Then generate results_summary.md: every headline claim stated in
one sentence, with its number, its confidence interval, the figure/table it
comes from, and its caveat. I will write the paper from this file — it must
contain no claim that our experiments did not actually produce.
```

## 🔼 MULTIPLIER

### M9 · Reproducibility package ★ — half a day
`pip freeze > requirements.lock.txt`, the input hashes from M9a, and publish
code + derived catalogue with a DOI (Zenodo, free).

`experiments/log.csv` already ties every number to a git commit and config hash
— this finishes the chain. ISEF-level judges ask whether work reproduces; a
hash-verified, version-pinned package answers in one sentence.

## ⚠️ How NOT to write the motivation
**2024: 320 of 861 sites (37.2%) fall below 3.1536 Mm³/yr.**

Those sites **are in the EOG record**. They are not undetected flares — so this
can't be phrased as "look how many VIIRS misses," because the data source *is*
VIIRS. And they are **0.65% of 2024 volume**, so "37% of sites are small"
invites "and under 1% of the gas, so why care?"

**The real motivation** is Seymour's finding that **industry reported 2.2× more
flaring than VIIRS detected** — a gap in flares absent from the catalogue
entirely, which is exactly why Sentinel-2 is needed and EOG cannot answer it.

What 37% *honestly* supports: *the sub-threshold range is real and populated
even inside the official record, so the population beyond it is unlikely to be
empty.* A supporting observation, not the headline.

**SUCCESS CHECK:** `results_summary.md` reads like your paper's results
section, and every number traces to `experiments/log.csv`.

---
---

# PITFALLS TO WATCH (read this twice)

1. **GEE quotas** — batch server-side, cache chunks, never loop scenes in
   Python. 150 EECU-hours, and debugging against GEE instead of the cache is
   how they vanish.
2. **Coordinate jitter** — EOG coordinates move between years; the 750 m
   clustering in Step 2 exists for that reason.
3. **Leakage** — the deadliest error: any EOG-derived quantity inside detection
   features, or a site in both train and test. **It has already happened once
   in this repo** (`splits.py` read the per-year table instead of the clustered
   catalogue, fixed in `1b8c2ce`). The split protocol is non-negotiable.
4. **Desert false positives** — steel plants, cement kilns, gas processing look
   like flares. The Step 10 chip review is your defence; never publish the
   hidden-flare number without it. M3 makes it a measurement rather than a
   spot check.
5. **The circularity caveat** — EOG truth comes from VIIRS. Say it in the paper
   before a judge says it to you.
6. **Scope discipline** — Steps 1–10 are the project. 11–12 are the wow. A
   bulletproof 1–10 beats a shaky 1–12.
7. **Silent unit and code drift** — the two worst bugs found so far were both
   silent: a country code that changes in one year only, and a column that
   switches between percent and fraction. Neither raised an error; both
   corrupted results. **Verify every column's units and every join's row count
   after any new data arrives.**
8. **Verify sources before building on them** — the documented AWS bucket did
   not exist. Confirm real bytes come back before writing the module.
9. **Claude Code habits** — keep `CLAUDE.md` updated as decisions change; ask
   for tests on join and feature code; commit after every green run.

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
circularity. M3 defends the central claim. All three attack the same weakness:
ground truth derived from the sensor under study.

---

# WHAT "DONE" LOOKS LIKE

A measured baseline failure (fig04), a model that beats it with confidence
intervals (fig06), a map of flares nobody has catalogued (fig07), a monitoring
system with a historical replay, a methane extension (finding or honest null),
an interactive app, and a results file where every sentence is backed by a
logged experiment. That is a top-tier Ebdaa project — and genuinely useful
science.

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
Stages: `check-env · download · join · splits · baseline · figures`

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
confidence interval · **Conformal** — a calibration step that makes intervals
achieve their stated coverage · **Confounding** — two explanations tangled so
one number can't separate them · **KS** — Kolmogorov–Smirnov, largest gap
between two cumulative distributions · **LORO** — leave-one-region-out ·
**GroupKFold** — cross-validation where a whole group (a site) stays on one
side of the split · **Holdout** — data withheld from training · **Leakage** —
test information reaching the model · **Complete linkage** — clustering that
caps cluster diameter · **SHAP** — per-feature attribution of a model's
prediction · **SWIR** — short-wave infrared, where flames glow (B11 1610 nm,
B12 2190 nm) · **SCL** — Sentinel-2 scene classification band, used for cloud
masking · **VIIRS** — the nighttime sensor behind the official record ·
**VNF** — VIIRS Nightfire, the per-detection product carrying radiant heat ·
**EECU-hour** — Earth Engine compute unit; you have 150 · **Radiant heat** —
power radiated by the flare; EOG's actual calibration input, absent from the
public per-site sheets · **MMBtu** — energy unit used for gas pricing.
