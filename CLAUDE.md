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

## VNF LICENCE CONSTRAINTS (read before designing any output)
VNF is covered by the VIIRS Nightfire Academic Data Use License (v.
2026-01-26). It is NOT click-through: it must be signed and emailed to
eog@mines.edu / victoria.patti@mines.edu, and approved. One year, no cost.
Five clauses change what this project may produce:

1(b) NO redistribution of VNF data in machine-readable form (csv/json/xml)
     or any format enabling bulk download.
     -> The reproducibility package may publish DERIVED products only.
        Never the raw VNF, and never a downloadable per-detection table.

1(e) NO public disclosure of flared gas volumes or CO2 emissions for any
     year EOG has not yet published.
     -> Our range 2019-2024 is cleared (EOG has published through 2024).
        2025 is EMBARGOED. Check the published-years list before releasing
        any volume or CO2e figure:
        https://eogdata.mines.edu/products/vnf/global_gas_flare.html

1(f) NO temporal profiles finer than WEEKLY, except:
       i.  internal use within the licensee's affiliation
       ii. static form -- scientific publication, presentation, prints,
           in non-downloadable formats
       iii.interactive web service for a RESTRICTED REGION, and only with
           EOG approval obtained BEFORE the app goes public
     -> Per-pass intermittency analysis is fine internally and fine as a
        static figure in the paper. The Streamlit app showing nightly
        per-site time series needs EOG's prior written approval, or must
        aggregate to weekly. Decide this before building the app, not after.

1(c) + 2(a) EVERY product, including graphs, carries:
     "This product was made utilizing VIIRS Nightfire (VNF) nightly data
      produced by the Earth Observation Group, Payne Institute for Public
      Policy, Colorado School of Mines."
     Short form for single images: "Source: VIIRS Nightfire, Colorado
     School of Mines." Logos: https://eogdata.mines.edu/products/logo/

1(g) + 2(b) Publications MUST cite EOG papers. The two that matter here:
     - Zhizhin et al. 2025, "An Improved Calibration for Satellite
       Estimation of Flared Gas Volumes from VIIRS Nighttime Data",
       Energies 18(17):4765 -- THE CURRENT CALIBRATION. Read before
       finishing baseline.py; it is the formula we are trying to reproduce.
     - Zhizhin et al. 2026, "VIIRS Nightfire Super-Resolution Method for
       Multiyear Cataloging of Natural Gas Flaring Sites: 2012-2025",
       Remote Sensing 18(2):314 -- EOG's own cross-year site cataloguing.
       Compare against our 750 m clustering.
     Plus Elvidge et al. 2013 (VNF description) and 2016 (flaring methods).

## Useful discoveries from the licence document
- GIREE, the Global Infrared Emitter Explorer:
  https://eogmap.mines.edu/giree
  A catalogue of ~20,000 industrial infrared emitters WITH EMITTER TYPE
  (steel mills, upstream flares, etc.) and nightly temporal profiles.
  This is very likely the ready-made NEGATIVE CONTROL SET for the
  false-positive audit: known industrial hot sources that are NOT flares
  are exactly what the detector must not fire on. Investigate before
  hand-building control points.
- EOG contacts: eog@mines.edu, victoria.patti@mines.edu,
  Dr Mikhail Zhizhin (VNF lead programmer) mzhizhin@mines.edu,
  Dr Chris Elvidge (EOG director) celvidge@mines.edu

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

## Which split supports which claim (measured 2026-08-31, do not mix these up)
Not every holdout can carry every claim. Measured on catalog.parquet
(11,787 site-years, 2,490 clustered sites, 2012-2024). KS = Kolmogorov-
Smirnov statistic between train and test log-volume distributions;
above 0.150 the pooled number cannot separate a size effect from the
effect being claimed.

  split            test n   KS raw   KS matched   verdict
  by-site           2,355    0.045      0.016     CLEAN
  by-year           1,647    0.209      0.049     confounded
  region, -IRN      4,487    0.195      0.020     confounded
  region, -IRQ      2,656    0.264      0.035     confounded
  region, -SAU      4,644    0.362      0.044     confounded, very-large unusable

So:
- BY-SITE is the only unconfounded split. It carries the HEADLINE
  performance numbers ("MAE X +/- Y"), and it is the only one that can.
- BY-YEAR carries the forward-monitoring claim, but it is confounded
  (KS 0.209): the size distribution shifts between the training years and
  2023-24, so it MUST be reported per size bin or size-matched. Do not
  quote a pooled by-year number.
- REGION folds carry the transfer claim ONLY, always with matching.
- Never let one number do both jobs.

EVERY region fold is confounded. There is no clean region comparison, so
every cross-country statement needs the matched set. Size-matching brings
all folds to KS <= 0.049, at the cost of sample size.

WHY: holding out a region changes the flare SIZE distribution at the same
time as the geography. Saudi median is 91 m3/h against 1,147 for Iraq+Iran.
A drop on the Saudi fold has two competing explanations - the model does
not travel, or the model is worse on small flares - and the pooled number
cannot separate them. src/splits.py quantifies this for every split and
builds a size-matched training set.

Leave-one-region-out, not one fixed fold: three folds give three transfer
tests, and a failure appearing in EVERY fold regardless of which region is
held out is about size, not about crossing a border.

STANDING LIMITATION for the write-up: no claim about very-large flares
(>20,000 m3/h) in Saudi Arabia is supportable - the catalogue holds 4 such
sites, below the 20-site floor.

CORRECTION HISTORY (keep, so the numbers are not silently re-trusted):
earlier drafts of this section reported by-year as unconfounded (KS 0.125)
and the -IRN fold as clean (KS 0.102). Both were computed before the Saudi
ISO-code bug was fixed, on a catalogue missing 12 years of Saudi data, and
on eog_site_year.parquet whose per-year ids made the by-site split leak.
Superseded by the table above.

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
