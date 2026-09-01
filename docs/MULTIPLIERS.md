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
