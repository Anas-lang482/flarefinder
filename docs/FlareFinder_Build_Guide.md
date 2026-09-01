# FlareFinder — Complete Build Guide
## From zero to a working AI model + application prototype
### "Seeing the Invisible Waste: Detecting Undetected Gas Flares Across the Gulf"

This guide takes you from an empty laptop to a trained, validated AI model with an interactive application, in 9 phases matching your 8-week plan. Every phase has: the goal, the steps, a **ready-to-paste Claude Code prompt**, and a success check so you know you're done.

---

## HOW TO USE THIS GUIDE

1. First, set up your project folder and paste the **Project Context Prompt** (below) into a file called `CLAUDE.md` in your project root. Claude Code automatically reads this file — it makes every future prompt smarter because Claude Code will always know the full project.
2. Work through phases in order. Paste each phase prompt into Claude Code when you reach it.
3. After each phase, run the success check. Don't move on until it passes.
4. Commit to git after every working step (`git add -A && git commit -m "phase X done"`).

---

## THE PROJECT CONTEXT PROMPT (put this in CLAUDE.md)

```markdown
# PROJECT: FlareFinder — AI Detection of Undetected Gas Flares in the Gulf

## Who I am
I am a high school student building a research project for the Ebdaa science
competition (Saudi Arabia, feeds into Regeneron ISEF). I have 8 weeks. I am
supervised by a mentor. Rigor and honesty matter more than impressive-looking
numbers: every result must survive expert judging.

## The science
When oil is extracted, associated gas is burned off at flare stacks. In 2024,
~151 billion m3 was flared globally (~$63B value, 389 Mt CO2e emissions
including 46 Mt unburnt methane). The official global flare record comes from
the VIIRS Nightfire satellite product, but:
- Seymour et al. 2025 (Environ. Sci. Technol.) proved VIIRS misses flares
  below ~360 m3/h; industry reported 2.2x more flaring than VIIRS detected.
- Volume estimates use hand-fitted calibration formulas with documented
  size-dependent bias (Elvidge et al. 2024, Energies: small flares
  overestimated, large underestimated).
- Sentinel-2 SWIR (20 m) can see flares VIIRS misses, but has only been
  operationalized OFFSHORE (Liu et al. 2023, Nature Sustainability); onshore
  desert ground creates false positives and thresholds don't transfer
  between regions.
- ML sensor fusion against metered volumes was demonstrated at ONE Iranian
  site (Asadi-Fard et al. 2024, JGR Atmospheres).

## My contribution (never overclaim beyond this)
The first ML system that (1) detects small ONSHORE flares in the Gulf by
fusing VIIRS/Black Marble nighttime thermal + Sentinel-2 daytime SWIR,
using the TEMPORAL INTERMITTENCY of flares (they flicker across passes;
hot ground follows the sun) to kill desert false positives; (2) replaces
hand-fitted volume calibrations with a learned, physics-constrained model
(volumes >= 0) with calibrated uncertainty on every estimate; (3) runs as
an automated monitoring pipeline + public interactive map of flares
invisible to the official record.

## Study region
Saudi Arabia, Iraq, Iran (onshore flare sites from the EOG catalog).

## Ground truth and its known weakness (be honest about this in all outputs)
EOG/Colorado School of Mines annual per-site flared volume estimates
(free download). CAVEAT: EOG volumes are themselves derived from VIIRS,
the sensor whose blind spots we target. So: EOG serves as volume reference
for sites it contains; Sentinel-2 provides independent evidence for sites
it does NOT contain. State this distinction in every analysis.

## Data sources (all free)
- EOG annual flare site spreadsheets: eogdata.mines.edu (account registered)
- VIIRS Nightfire / Black Marble combustion: AWS open data
  (s3://blackmarble-combustion, --no-sign-request) and EOG VNF files
- Sentinel-2 L2A SR: Google Earth Engine, COPERNICUS/S2_SR_HARMONIZED
  (SWIR bands B11 1610nm, B12 2190nm; flames glow in SWIR)
- Sentinel-5P methane: GEE COPERNICUS/S5P/OFFL/L3_CH4 (extension only)

## Method rules (non-negotiable)
1. NEVER evaluate on training sites/years. Holdouts: by-site AND by-year.
2. Report error AS A FUNCTION OF FLARE SIZE (small flares are the point).
3. Every headline number gets a bootstrap confidence interval.
4. Volume model must be structurally unable to predict negative volumes.
5. Prefer the simpler model when performance is equal (GBT before CNN).
6. Baselines FIRST: reproduce standard VIIRS calibration before building
   anything new. We must measure the failure we claim to fix.
7. All code reproducible: fixed seeds, config files, no magic numbers.
8. Log every experiment (params + metrics) to experiments/log.csv.

## Stack
Python 3.11+. earthengine-api, geemap, pandas, geopandas, numpy,
scikit-learn, xgboost, lightgbm, matplotlib, shap, folium.
PyTorch only if/when we add the CNN. App: Streamlit + folium map.

## Repo layout
flarefinder/
  CLAUDE.md            (this file)
  config.yaml          (region bounds, years, thresholds, seeds)
  data/raw/            (EOG downloads — never modified)
  data/processed/      (parquet tables the pipeline builds)
  src/                 (one module per pipeline stage)
  notebooks/           (exploration only, nothing load-bearing)
  experiments/log.csv  (every run: date, config, metrics)
  models/              (saved models)
  app/                 (Streamlit prototype)
  figures/             (publication-quality outputs)

## Tone for all outputs
Honest, precise, uncertainty-first. "We estimate X ± Y" not "X". When a
result is negative or weak, say so plainly — a measured negative result
is a contribution.
```

---

## PHASE 0 — Environment & accounts (Day 1)

**Goal:** working environment, all accounts requested, empty repo with structure.

**You do manually (not Claude Code):**
1. Register: Google Earth Engine (noncommercial), NASA Earthdata, EOG (eogdata.mines.edu).
2. Install: Python 3.11+, git, VS Code, Claude Code.
3. `earthengine authenticate` once GEE approves you.

**Claude Code prompt:**
```
Set up the project skeleton exactly as described in CLAUDE.md: create the
repo layout, a config.yaml with placeholders for region bounds (Saudi
Arabia, Iraq, Iran bounding boxes), years (2019-2025), random seed 42,
and a requirements.txt with the stack listed in CLAUDE.md. Create a
Makefile or simple run.py with commands for each future pipeline stage
(download, join, features, baseline, train, evaluate, app). Initialize
git with a sensible .gitignore for data/ and models/. Then write
src/check_env.py that verifies every library imports and that Earth
Engine authenticates, printing PASS/FAIL per item. Run it and fix
whatever fails.
```

**Success check:** `python src/check_env.py` prints all PASS.

---

## PHASE 1 — Ground truth: the EOG flare catalog (Days 2–3)

**Goal:** a clean table of every known flare site in Saudi Arabia, Iraq, Iran with coordinates and annual volumes — your study population and regression targets.

**You do manually:** download the EOG "Global Gas Flaring" annual per-site .xlsx files (as many years as available, ideally 2019–2024) into `data/raw/eog/`.

**Claude Code prompt:**
```
In src/build_catalog.py: load every EOG annual flare-site xlsx in
data/raw/eog/. These files have per-site rows with lat, lon, country, and
estimated flared volume (bcm or million m3 — check units per file and
normalize to million m3/year, documenting the conversion). Filter to
Saudi Arabia, Iraq, Iran. Deduplicate sites across years by clustering
coordinates within 750 m (flare coordinates jitter between years) and
assign a stable site_id. Output data/processed/catalog.parquet with:
site_id, lat, lon, country, year, volume_mcm, n_years_observed.
Print a summary: sites per country, volume distribution (log-histogram),
and how many sites fall below the Seymour small-flare threshold
(~360 m3/h ≈ 3.15 million m3/year if continuous — compute and state the
assumption). Then build figures/fig01_flare_map.html: a folium map of
all sites, marker size by volume, colored by country. Also export a
static PNG version for the paper.
```

**Success check:** catalog.parquet exists; the map shows hundreds of sites clustered in known oil regions (Eastern Province, southern Iraq, Khuzestan/Persian Gulf coast of Iran). Note the small-flare count — that number goes in your paper's motivation.

---

## PHASE 2 — Satellite signals for every site (Week 2)

**Goal:** for each site (and matched non-flare control points), extract (a) VIIRS/Black Marble nighttime detections and (b) Sentinel-2 SWIR statistics over time. This is the dataset everything else trains on.

**Claude Code prompt (part A — controls & sampling design):**
```
In src/make_controls.py: for each flare site in catalog.parquet, generate
3 control points 5-15 km away that are NOT within 2 km of any known flare
site (any country, any year). Controls must sample the same desert/land
surface, so keep them onshore (use a land mask via GEE or natural-earth
polygons). Output data/processed/controls.parquet with control_id, lat,
lon, paired_site_id. Print counts and add controls to the folium map in
a second layer (gray markers).
```

**Claude Code prompt (part B — Sentinel-2 SWIR extraction via GEE):**
```
In src/extract_s2.py: using the earthengine-api with batched server-side
reduceRegions (never per-image Python loops — design for ~1500 points x
~6 years), extract for every site and control point a monthly time series
from COPERNICUS/S2_SR_HARMONIZED: mean/max of B11 and B12 in a 100 m
buffer, the normalized SWIR ratio (B12-B11)/(B12+B11), cloud fraction
(from SCL band), and observation count. Handle GEE quota limits with
chunking and exponential backoff retries; cache completed chunks to
data/processed/s2_chunks/ so the job resumes if interrupted. Merge into
data/processed/s2_timeseries.parquet keyed by (point_id, year, month).
Validate: for the 20 largest-volume sites, plot B12 max vs time — we
expect persistently elevated SWIR at big flares. Save as
figures/fig02_s2_sanity.png. If big flares do NOT show elevated SWIR,
STOP and tell me — that invalidates the approach and we must debug
(buffer size, cloud masking, or coordinate issues) before continuing.
```

**Claude Code prompt (part C — nighttime combustion extraction):**
```
In src/extract_vnf.py: pull the Black Marble / VIIRS nighttime combustion
detections for our region and years. Primary source: the AWS open bucket
(s3://blackmarble-combustion or the registry-listed path, accessed with
--no-sign-request via boto3 anonymous config). If schema differs from
expectation, inspect and adapt, documenting the actual schema in a
comment. For every site/control and month, compute: number of nighttime
detections within 750 m, mean/max radiant heat if available. Output
data/processed/vnf_timeseries.parquet. Sanity check: detection counts at
known large flares should be far higher than at controls — print that
comparison as a table.
```

**Success check:** two parquet time-series tables; sanity figures show big flares glowing in SWIR and lighting up at night, controls dark in both.

---

## PHASE 3 — Baselines: reproduce the failure you claim to fix (Week 3)

**Goal:** implement the standard VIIRS volume calibration + measure where it fails, by flare size. This plot is the scientific heart of your motivation.

**Claude Code prompt:**
```
In src/baseline.py: implement the standard VIIRS Nightfire volume
calibration (Elvidge-style: flared volume as a function of radiant heat,
using the published functional form and coefficients — cite them in
comments; if the exact published coefficients are ambiguous from the
papers we have, fit the SAME functional form on our training years and
say so explicitly). Apply it to our VNF signals to produce per-site
annual volume estimates. Compare against EOG catalog volumes on held-out
YEARS (train coefficients on 2019-2022, evaluate 2023+). Produce:
(1) figures/fig03_baseline_scatter.png: predicted vs EOG volume, log-log,
colored by size class; (2) figures/fig04_error_vs_size.png: relative
error binned by true volume decile with bootstrap 95% CIs — this is the
headline motivation figure; (3) detection-side table: what fraction of
EOG sites in each size decile have ZERO nighttime detections in a given
year (the invisibility rate). Write results to experiments/log.csv.
Report honestly: our claim is that error and invisibility are worst for
small flares — check whether OUR data shows that. If it does not, flag
it loudly.
```

**Success check:** error-vs-size figure exists and (expected) shows degradation at small volumes. Whatever it shows, you now have a measured baseline — the thing your model must beat.

---

## PHASE 4 — The detection model (Weeks 4–5, part 1)

**Goal:** classify flare vs. non-flare from fused features, with the flicker/intermittency signature. Headline metric: recall on SMALL flares vs. nighttime-only detection.

**Claude Code prompt:**
```
In src/features.py then src/train_detector.py:

FEATURES per point-year, from the two time-series tables: SWIR level
stats (mean/max/p95 of B11, B12, ratio), TEMPORAL features that encode
flicker — coefficient of variation of monthly B12 max, fraction of months
with B12 above a locally-learned percentile threshold, lag-1
autocorrelation (hot ground is seasonally smooth and sun-driven; flares
switch on/off), day-night agreement (S2 elevated AND/OR VNF detections),
VNF detection count and radiant heat stats, and site context (distance
to nearest known flare cluster). No leakage: nothing derived from EOG
volume enters detection features.

MODEL: gradient-boosted trees (LightGBM), binary flare vs control.
SPLITS: GroupKFold by site_id (a site never appears in both train and
test) AND a temporal split (train <=2022, test 2023+). Report both.
METRICS: overall ROC-AUC and PR-AUC, but the HEADLINE is recall at 95%
precision, broken out by EOG volume decile — especially the smallest
deciles and sites with zero VNF detections (the invisible ones).
Compare against the nighttime-only rule (VNF detections > 0) as the
competitor. Ablation: with vs without the temporal flicker features —
we claim flicker kills desert false positives; measure precision impact.
Bootstrap CIs on all headline numbers. SHAP summary plot to
figures/fig05_detector_shap.png; results table to
figures/tab01_detection.csv and experiments/log.csv.
```

**Success check:** detector beats the nighttime-only rule on small-flare recall at matched precision, with CIs. The flicker ablation shows a measurable precision gain (if not — that's a finding; report it honestly).

---

## PHASE 5 — The volume model (Weeks 4–5, part 2)

**Goal:** learned calibration: fused features → annual volume, physics-constrained, uncertainty on every estimate. Must beat the Phase 3 baseline, especially on small flares.

**Claude Code prompt:**
```
In src/train_volume.py: regression from fused features (Phase 4 features
+ VNF radiant heat stats) to EOG annual volume, on EOG-cataloged sites.

PHYSICS CONSTRAINT: model log(volume) so predictions are structurally
positive; document this in comments and paper notes.
MODELS: LightGBM quantile regression (q10/q50/q90) as primary; optional
small PyTorch MLP as ablation only if LightGBM plateaus.
UNCERTAINTY: calibrate the q10-q90 interval with split-conformal
adjustment on a calibration fold so empirical coverage on held-out data
is ~80%; report achieved coverage overall AND for the smallest size
decile separately (coverage often breaks exactly where it matters —
measure it, don't assume it).
SPLITS: same protocol as Phase 4 (by-site GroupKFold + temporal).
EVALUATION: relative error vs size deciles with CIs, side by side with
the Phase 3 baseline on IDENTICAL held-out data —
figures/fig06_volume_vs_baseline.png is the single most important figure
of the project. Also report transfer: train on 2 countries, test on the
third, as a table. Log everything to experiments/log.csv.
```

**Success check:** your model's error-vs-size curve sits below the baseline's, with non-overlapping CIs at least in the small deciles; conformal coverage within a few points of nominal. If the model does NOT beat the baseline by end of Week 6 → execute the proposal's fallback (methane track becomes primary; the negative result gets reported honestly).

---

## PHASE 6 — The hidden-flare estimate (Week 6)

**Goal:** the number nobody has: how much Gulf flaring is invisible to the official record.

**Claude Code prompt:**
```
In src/hidden_flares.py: run the trained detector over a systematic grid
of candidate locations in the study region's oil zones (define zones as
a buffered union around known flare clusters plus known oil-field
polygons if available from natural-earth/OSM; document the definition).
Flag high-confidence detections that are NOT in the EOG catalog
(>=1.5 km from any cataloged site). For each, estimate volume + conformal
interval with the volume model. Aggregate: total hidden volume per
country with propagated uncertainty (sum of intervals via bootstrap over
sites). Convert to dollars (state gas price assumption as a RANGE,
e.g., $2-8/MMBtu → document conversion m3→MMBtu) and CO2e (state
emission factor source). Output data/processed/hidden_flares.parquet and
figures/fig07_hidden_map.html (red = hidden, gray = cataloged).
CRITICAL honesty pass: manually review the top 20 hidden detections in
Sentinel-2 imagery (write a helper that exports S2 image chips per site
to figures/chips/) — industrial heat sources like steel plants or gas
processing can masquerade as flares. Label each chip plausible-flare /
industrial / unclear, and report the plausibility rate alongside the
headline number.
```

**Success check:** a hidden-flare map + a headline sentence like "we detect N probable flares absent from the official record, totaling X ± Y million m3/year (~$A–B million, ~C kt CO2e)" — with the manual-review plausibility rate attached.

---

## PHASE 7 — Monitoring pipeline + methane extension (Week 7)

**Goal:** turn the models into a system: process new Sentinel-2 passes automatically, flag changes; estimate unburnt methane at detected sites.

**Claude Code prompt (monitoring):**
```
In src/monitor.py: build a run_monitor(date_range) function that queries
GEE for new S2 scenes over the study region since the last run
(state stored in data/processed/monitor_state.json), recomputes features
for all tracked + hidden sites, runs the detector, and emits a change
report: NEW detections, sites gone dark, volume-trend flags. Output both
a machine-readable JSON and a human-readable markdown report to
reports/. Add a --simulate mode that replays 2024 month by month to
demonstrate the system on historical data (this becomes the demo for
judges: 'here is January… here a new flare appears in March').
```

**Claude Code prompt (methane extension — only if Phases 4-6 are solid):**
```
In src/methane.py: for our detected sites (cataloged + hidden), extract
Sentinel-5P L3 CH4 (GEE COPERNICUS/S5P/OFFL/L3_CH4) monthly means in
site-centered windows vs matched upwind background windows, and test for
site-associated enhancement. Be conservative and honest: S5P pixels are
~7 km, far coarser than flares, so individual-site attribution is weak —
frame results as regional/cluster-level enhancement, with significance
testing (paired test vs background, multiple-comparison correction).
Where the openly published labeled plume datasets (Schuit 2023 Zenodo,
NASA EMIT plume products) contain plumes in our region, cross-reference:
do any coincide with our hidden detections? Report as a table with clear
caveats. This is the EXTENSION — if signals are too weak, say so; a
careful null result here does not harm the core project.
```

**Success check:** `python src/monitor.py --simulate 2024` produces monthly change reports; methane section produces either a cautious positive finding or an honest null.

---

## PHASE 8 — The application prototype (Week 7–8)

**Goal:** the public-facing artifact — an interactive app judges can touch.

**Claude Code prompt:**
```
In app/: build a Streamlit application "FlareFinder" with:
1. MAP page (main): folium map of the Gulf. Layers: cataloged flares
   (gray), hidden flares we detected (red), controls off by default.
   Click a site → sidebar shows its S2/VNF time series plot, model
   volume estimate with the conformal interval, detection confidence,
   and the S2 image chip. Filters: country, size class, year,
   hidden-only toggle.
2. IMPACT page: headline counters computed live from
   hidden_flares.parquet — hidden sites found, hidden volume/year with
   interval, dollar range, CO2e — each with a "how computed" expander
   stating every assumption (gas price range, emission factors,
   continuity assumption).
3. MONITOR page: renders the latest reports/ markdown from Phase 7 and
   a month-slider replay of the 2024 simulation.
4. METHODS page: honest one-screen summary — data sources, the EOG
   circularity caveat, validation protocol, and limitations, written
   for a non-expert judge.
Design: clean, fast, works offline from local parquet files (no live
GEE calls in the app). Arabic/English toggle for headline text if
straightforward. Add app/README.md with one-command run instructions
(streamlit run app/Home.py). Test that it runs end to end with our
actual data files.
```

**Success check:** the app runs locally, a judge (or your mentor) can click a red dot and see *why* the model believes it's a flare. This is your demo.

---

## PHASE 9 — Results, figures, paper (Week 8)

**Goal:** consolidate everything into the Ebdaa paper + poster material.

**Claude Code prompt:**
```
In src/make_paper_assets.py: regenerate all publication figures at 300
DPI with consistent styling and numbered filenames matching this list:
fig01 study-region flare map, fig04 baseline error-vs-size, fig05
detector SHAP, fig06 volume model vs baseline (the headline), fig07
hidden-flare map (static PNG export), plus tab01-tab04: detection
metrics, volume metrics with coverage, country transfer, hidden-flare
totals with assumptions. Then generate results_summary.md: every
headline claim of the project stated in one sentence each, with its
number, its confidence interval, the figure/table it comes from, and
its caveat. I will write the paper from this file — it must contain no
claim that our experiments did not actually produce.
```

**Success check:** `results_summary.md` reads like your paper's results section, and every number traces to `experiments/log.csv`.

---

## PITFALLS TO WATCH (read this twice)

1. **GEE quotas** — batch server-side, cache chunks, never loop scenes in Python.
2. **Coordinate jitter** — EOG site coordinates move slightly between years; the 750 m clustering in Phase 1 exists for that reason.
3. **Leakage** — the deadliest error: any EOG-derived quantity inside detection features, or a site appearing in both train and test. The split protocol in Phases 4–5 is non-negotiable.
4. **Desert false positives** — steel plants, cement kilns, gas processing look like flares. The Phase 6 manual chip review is your defense; never publish the hidden-flare number without it.
5. **The circularity caveat** — EOG truth comes from VIIRS. Say it in the paper before a judge says it to you.
6. **Scope discipline** — Phases 1–6 are the project. Phases 7–8 are the wow. If time runs short, a bulletproof 1–6 beats a shaky 1–8.
7. **Claude Code habits** — keep CLAUDE.md updated as decisions change; ask it to write tests for the join and feature code (silent data bugs are the ones that kill projects); commit after every green run.

---

## WHAT "DONE" LOOKS LIKE

By the end you will have: a measured baseline failure (fig04), a model that beats it with confidence intervals (fig06), a map of flares nobody has cataloged (fig07), a monitoring system with a historical replay, a methane extension (finding or honest null), an interactive app, and a results file where every sentence is backed by a logged experiment. That is a top-tier Ebdaa project — and genuinely useful science.

---
---

# POTENTIAL MULTIPLIERS

*Added 2026-09-01, after Phases 0–1 and an early Phase 3 baseline were built.*

A multiplier is not "more work" — it is work that **raises the ceiling on what
the project can claim**. Ordered by impact per hour. Each says exactly where
it slots into the phase plan.

Read M0 first: it is the constraint the top three all address.

---

## ⚠️ M0 — THE STRUCTURAL WEAKNESS THESE ARE FIXING

The project's ground truth (EOG volumes) is produced **by VIIRS**, the sensor
whose blind spots the project targets. Consequences already measured:

- The Phase 3 baseline could not reproduce EOG's real calibration — radiant
  heat and source area are absent from the public per-site sheets, so what
  got fitted was a detection-frequency regression (R² 0.831 from that one
  variable; temperature contributes +0.007).
- Sub-threshold sites in the record are **0.65% of 2024 volume**, so "we found
  small flares" invites "and they don't matter."

**Every multiplier below either supplies independent evidence, or converts
findings into a unit that carries weight.** That is the whole game.

---

## M1 — VNF per-detection records ★★★ START BEFORE PHASE 2

**What:** VIIRS Nightfire per-overpass records: radiant heat, source
temperature, source area, timestamp — per detection, not per year.

**Why it multiplies:**

1. **Unlocks a real Phase 3.** With radiant heat you can apply EOG's actual
   published slope (0.029353) and genuinely reproduce their calibration.
   Today's baseline carries a caveat that removes most of its force.
2. **Turns intermittency from annual into per-pass.** `detection_freq` is one
   number per site-year. VNF gives the actual time series — which *is* the
   flicker signature your false-positive defence depends on. This is the
   single biggest upgrade available to your core method.
3. Enables sub-annual analysis: seasonality, shutdowns, sites appearing
   mid-year.

**Blocking issue:** `s3://blackmarble-combustion` does not exist. Two live
candidates: EOG VNF files at `eogdata.mines.edu` (your account already
works), or NASA Black Marble VNP46 via LAADS (needs the Earthdata token).

**Cost:** 1–2 days, mostly download and parse. No new skills.

**Do it:** immediately. It is already blocking, and Phase 2 features are
better designed once you know what VNF actually gives you.

---

## M2 — Independent reported volumes ★★★ DURING PHASE 2, BEFORE PHASE 5

**What:** Country- or operator-reported flaring volumes that do **not** come
from satellites. Candidates: World Bank GGFR / GFMR country tables, Saudi
Aramco sustainability reports, Iraq Ministry of Oil / Basra Gas Company
statements, NIOC reporting, and the reported figures Seymour et al. 2025 used.

**Why it multiplies:** It breaks the circularity in M0. Right now you can only
ever say "our model agrees with EOG." With an independent reference you can
say **"the satellite record under-reports relative to what operators
themselves report, by X%, and our system recovers Y% of that gap."** That is a
fundamentally stronger sentence, and it is the sentence Seymour's 2.2× result
licenses you to test.

Even country-level totals help: aggregate your per-site estimates to a
national figure and compare against reported. Real external validation, no
site-level metering required.

**Honest limit:** reported figures are self-reported and carry their own
biases. Say so. A comparison against two imperfect references is still far
better than one circular one.

**Cost:** 1–2 days of document hunting. No code.

**Do it:** during Phase 2 while GEE extractions run. **Must exist before
Phase 5**, or the volume model has nothing external to be judged against.

---

## M3 — Negative control / false-positive audit ★★★ WITH PHASE 4

**What:** Run the finished detector over areas with **no known flares** —
empty desert, quarries, steel plants, cement kilns, urban heat, solar farms.
Measure the false-positive rate. Report it as a headline number.

**Why it multiplies:** Your central claim is "we detect onshore flares that
VIIRS misses." The obvious attack is: *"how do you know those aren't hot
desert or an industrial furnace?"* The Phase 6 manual chip review is a spot
check; a systematic negative control is a **measurement**. It converts your
weakest point into a quantified strength.

This is also the single most likely reason a judge disbelieves the final map.

**Cost:** ~1 day. Sample N random non-flare locations stratified by land
cover, run the pipeline, count detections.

**Do it:** as part of Phase 4, not after. Treat it as non-optional — the
detection model is not finished until this number exists.

---

## M4 — CO₂e and unburnt-methane conversion ★★ AFTER PHASE 6

**What:** Convert detected volumes into CO₂-equivalent emissions and unburnt
methane, using published combustion-efficiency figures (flares are typically
assumed ~92–98% efficient; the unburnt remainder is largely methane, ~28×
CO₂ over 100 years).

**Why it multiplies:** It answers the "0.65% of volume, so what?" objection
directly. "We found 400 uncatalogued flares" is a satellite result. **"We
found 400 uncatalogued flares representing X kt of unaccounted CO₂e, of which
Y kt is unburnt methane"** is a climate result — and methane is where the
policy urgency sits, because a small unburnt volume carries disproportionate
warming.

**Honest requirement:** carry your volume uncertainty through the conversion.
Report a range, and state the efficiency assumption as an assumption.

**Cost:** half a day. Arithmetic plus a literature value.

**Do it:** immediately after Phase 6 produces the hidden-flare estimate, so
Phase 9 leads with a climate number rather than a count.

---

## M5 — Landsat 8/9 as a third sensor ★★ AFTER PHASE 6, OPTIONAL

**What:** Add Landsat 8/9 SWIR (30 m, OLI) alongside Sentinel-2.

**Why it multiplies:** More overpasses means a better-resolved flicker
signature — and intermittency is your discriminator. A different overpass time
also gives a partly independent look, so agreement between two independent
daytime sensors is much harder to dismiss than one.

**Cost:** ~1 day; Landsat is already in Earth Engine and `data_s2.py` will
generalise. Watch the EECU budget.

**Do it:** only if Phases 1–6 are solid. A strengthener, not a requirement.

---

## M6 — Calibrated uncertainty, not just intervals ★★ WITH PHASE 5

**What:** Check that your prediction intervals are **actually calibrated** — a
95% interval should contain the true value 95% of the time. Report measured
coverage per size bin. `config.yaml` already lists
`coverage_of_prediction_interval` as a metric.

**Why it multiplies:** Almost every student ML project reports accuracy. Very
few report whether their stated uncertainty is *honest*. "Our 95% intervals
achieve 94.2% coverage on held-out sites" signals real statistical maturity —
and if coverage comes out at 60%, you have found something important about
your own model before a judge does.

**Cost:** a few hours once the quantile model exists.

**Do it:** inside Phase 5. It is a property of the volume model, not an add-on.

---

## M7 — Report the data-quality findings to EOG ★★ DO NOW

**What:** Write a short, precise note to the EOG / Payne Institute team
(Elvidge, Zhizhin) documenting what this project found in their public files:
the Saudi ISO code changing to `SAUCROP` in 2024 only, the `detection_freq`
percent/fraction inconsistency across years, and the annual ID re-issue.

**Why it multiplies:**

- Each is a **real, reproducible defect in a widely used public dataset**, and
  each silently corrupts downstream analyses. Other researchers are hitting
  these right now.
- A documented exchange with the dataset's authors is a strong credibility
  signal for ISEF — it shows the work reached practitioners.
- They may grant VNF access or clarify the calibration, which directly serves
  M1.

**Cost:** 1 hour. Be brief, factual, include reproduction steps.

**Do it:** now. Response latency is outside your control, so start the clock.

---

## M8 — Temporal trend against Zero Routine Flaring 2030 ★ DO NOW

**What:** You already hold 13 years (2012–2024). Ask: is flaring in these
three countries falling, flat, or rising? How does that compare with the
World Bank Zero Routine Flaring by 2030 commitment?

**Why it multiplies:** A policy-relevant result available **today**, from data
already on disk, with no new pipeline. It gives Phase 9 a second independent
finding and frames the project against a real international commitment rather
than only against a sensor limitation.

**Cost:** 2–3 hours. Group `catalog.parquet` by year and country, bootstrap
the trend, plot it.

**Watch out:** catalogue coverage changes across years (2019 has 7,209 rows
globally, 2024 has 10,690). A rising site count may be better detection, not
more flaring. Compare **volumes**, normalise by detection coverage, and state
which you are measuring.

**Do it:** now, as a break between phases. Cheap and self-contained.

---

## M9 — Reproducibility package ★ PHASE 9

**What:** Freeze `requirements.lock.txt`, record input file SHA-256 hashes,
publish code plus derived catalogue with a DOI (Zenodo is free).

**Why it multiplies:** Judges at ISEF level ask whether work can be
reproduced. Handing over a hash-verified, version-pinned package answers that
in one sentence. `experiments/log.csv` already ties every number to a git
commit and a config hash — this finishes the chain.

**Note:** the download SHA-256 hashes were printed when the EOG files were
fetched but are **not yet stored in the repo**. If EOG silently republishes a
workbook, your results shift with no warning. Storing those hashes is a
30-minute job worth doing early.

**Cost:** half a day.

**Do it:** Phase 9 — but store the input hashes now.

---

## WHERE THEY SLOT IN

```
NOW, in parallel with everything ......... M7 (email EOG)
                                           M8 (temporal trend)
                                           store input hashes (from M9)

BEFORE PHASE 2 ........................... M1 (VNF records)         ★★★

DURING PHASE 2 ........................... M2 (independent volumes) ★★★

WITH PHASE 4 ............................. M3 (negative control)    ★★★

WITH PHASE 5 ............................. M6 (interval coverage)

AFTER PHASE 6 ............................ M4 (CO2e conversion)     ★★
                                           M5 (Landsat, optional)

PHASE 9 .................................. M9 (repro package)
```

**If you only do three: M1, M2, M3.** M1 makes the baseline real, M2 breaks
the circularity, M3 defends the central claim. Everything else is upside.

**Scope discipline still applies.** The guide's own warning holds: a
bulletproof Phases 1–6 beats a shaky 1–8. A multiplier that destabilises the
core is not a multiplier. Add them in the order above, and drop any still
unfinished two weeks before the deadline.
