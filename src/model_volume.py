"""Stage 8 -- flared volume model.

Learned replacement for hand-fitted calibration formulas. RULE 4: structurally unable to predict a negative volume (log1p target, expm1 inverse -- not clipping). Emits quantiles, so every estimate carries an uncertainty band.

STATUS: stub. Nothing implemented yet.
All numeric parameters must come from config.yaml (RULE 7).
"""
