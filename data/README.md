# Data provenance

- **Source:** ULB / Worldline credit card fraud benchmark, mirrored by OpenML.
- **Pinned identifier:** OpenML dataset 1597, version 1; file 1673544.
- **Metadata:** <https://www.openml.org/api/v1/json/data/1597>
- **Raw artifact:** <https://www.openml.org/data/v1/download/1673544/creditcard.arff>
- **Publisher MD5:** `178bcf9bb1f31a3dfe12d0e577884add`.
- **Local path:** `data/raw/creditcard.arff`, unchanged from the source.
- **Provenance files:** `source-openml-1597.json` preserves provider metadata; `manifest.json` records retrieval time, size and checksums.
- **Source license label:** OpenML labels this release “Public”; this is not a substitute for a specific redistribution license. Keep the raw file out of Git and review publisher terms before public redistribution.

## Dataset meaning

The source describes European credit card transactions from September 2013: 284,807 observations and 492 frauds. `Class` is the label (1 = fraud). `V1`–`V28` are anonymized PCA components; `Time` is elapsed seconds from the first observation and `Amount` is transaction amount. `Time` is not a unique transaction identifier.

Citation supplied by OpenML: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi, “Calibrating Probability with Undersampling for Unbalanced Classification,” IEEE CIDM, 2015. Attribute the dataset creators and the ULB/Worldline source when publishing the project.

## Handling rules

1. Preserve the downloaded bytes in `raw/`; write conversions, cleaned tables and features elsewhere.
2. Keep the target out of inference inputs and fit transformations only within training splits.
3. Audit duplicate transactions and tied timestamps before defining chronological evaluation; do not assume elapsed seconds identify customers or transactions.
4. Record row-level split membership and the cutoff strategy when evaluation is implemented.
5. A two-day sample supports a bounded offline benchmark. Long-running drift, identity fraud performance, review effectiveness and actual company impact are outside its observed evidence.
6. Use source-derived sample inputs for the application when display is permitted. Any generated tests or economic scenarios are supplementary and labeled.

Run `python3 scripts/download_data.py --verify-only` from the project directory to recheck the local raw file without network access.
