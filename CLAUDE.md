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

STRETCH GOAL (only if the core Gulf onshore work is finished and validated):
extend to the wider Middle East, and to offshore as well as onshore.
This is an EXTENSION, not part of the core claim. Two honesty constraints:
- Offshore detection is already operationalized (Liu et al. 2023), so the
  novelty offshore is cross-region generalization of ONE model, not
  offshore detection itself. Do not present it as new capability.
- Any expansion must re-run a BY-REGION holdout (train on some countries,
  test on held-out ones). If the model does not transfer, say so — that is
  a publishable negative result and directly tests the known problem that
  Sentinel-2 thresholds don't transfer between regions.

## Ground truth and its known weakness (be honest about this in all outputs)
EOG/Colorado School of Mines annual per-site flared volume estimates
(free download). CAVEAT: EOG volumes are themselves derived from VIIRS,
the sensor whose blind spots we target. So: EOG serves as volume reference
for sites it contains; Sentinel-2 provides independent evidence for sites
it does NOT contain. State this distinction in every analysis.

## Data sources (all free)
- EOG annual flare site spreadsheets: eogdata.mines.edu (account registered)
- VIIRS Nightfire / Black Marble combustion:
  !! CORRECTION (verified 2026-08-31): s3://blackmarble-combustion DOES NOT
     EXIST. Probed anonymously with boto3 -> NoSuchBucket, as did seven
     plausible variants. Do not build against this path.
  Candidate replacements, in order of preference:
    1. EOG VNF files from eogdata.mines.edu (EOG account -- already held).
       Per-pass nighttime detections. VERIFY next.
    2. NASA Black Marble VNP46 via LAADS DAAC. Requires a NASA Earthdata
       login token -- this is why the Earthdata account matters after all.
  Whichever is used, confirm it returns real bytes BEFORE writing
  src/data_viirs.py against it.
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

## Environment
Windows 11. Shell commands are written for the Git Bash terminal in VS Code
(POSIX syntax: forward slashes, `$VAR`, `&&`, heredocs, `/dev/null`).
