"""Stage 2 -- VIIRS / Black Marble ingest.

Pulls nighttime combustion detections from the public S3 bucket and EOG VNF files. Produces per-pass detection records, not annual aggregates -- the per-pass detail is what the intermittency feature needs.

STATUS: stub. Nothing implemented yet.
All numeric parameters must come from config.yaml (RULE 7).
"""
