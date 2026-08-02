# VASU-140M likelihood training-exclusion registry

Status: **registry constructed and validated.**

The self-curated likelihood qualification reserves 1,024 whole parent documents
per candidate source. Even though that suite is not independent production
evidence, any future data builder must exclude those documents to prevent
internal evaluation leakage.

The versioned registry binds the two pending source-admission packages, all four
likelihood manifests, and the exact union of development and sealed held-out
parent IDs. Validation fails if a source package, manifest, provenance record,
parent ID, count, or hash changes. It preserves the pending and non-authorizing
state of both sources and creates no training data.

The frozen registry is
`configs/data/exclusions/vasu_140m_likelihood_self_curated_v1.json` with
identity `5e7e9ac7ec7db8aedd308849d713797762878f447b456abcfece026c9a7c8e25`.
It excludes exactly 1,024 FineWeb parents and 1,024 Wikimedia parents. Their
source-specific exclusion identities are respectively
`b4c21d0e8b940e3d34670cfa19b9246a63ef02615392fcbb20a87742e4af0f9f`
and `9ee5bd7afca0daa5ab0ce31c83485486bc8b9f66bd338e5e58844ad15d7df339`.
