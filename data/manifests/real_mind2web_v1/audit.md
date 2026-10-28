# Mind2Web Real-Data Split Audit

- Status: **ready**
- Source: `osunlp/Mind2Web` (train)
- Revision: `local-unpinned`
- License: `CC-BY-4.0`
- Verification: `official_train0_sha256_plus_local_sha256`
- Seed: `42`
- Manifest SHA-256: `8ef0c1b7844dc5d83371093b63f622ab058710cf0a9fac736b9ad6e11bb99a57`

## Corpus Inventory

- Unique tasks: 1009
- Websites: 73
- Recorded actions: 7775
- Actions with captured observations: 7775
- Unique observation snapshots: 7775
- Duplicate task IDs excluded: 0
- Cross-site duplicate snapshots: 0

## Split Summary

| Split | Tasks | Websites | Usable actions | Source actions | Missing observations |
|---|---:|---:|---:|---:|---:|
| train | 718 | 51 | 5607 | 5607 | 0 |
| val | 144 | 11 | 990 | 990 | 0 |
| test | 147 | 11 | 1178 | 1178 | 0 |

## Per-Split Annotation Coverage

| Split | Rationale | Positive candidates | Original target | Candidate geometry |
|---|---:|---:|---:|---:|
| train | 0 | 5328 | 3706 | 5321 |
| val | 0 | 926 | 647 | 926 |
| test | 0 | 1108 | 761 | 1103 |

## Categories

- domain: entertainment=261, shopping=281, travel=467
- subdomain: airlines=106, auto=57, car rental=43, department=15, digital=28, event=58, fashion=32, game=41, general=99, ground=72, hotel=32, movie=62, music=46, other=105, restaurant=63, speciality=96, sports=54

## Source Operations

- CLICK: 6513
- SELECT: 326
- TYPE: 936

## Annotation Coverage

- Actions with recorded rationale: 0
- Actions without recorded rationale: 7775
- Actions with positive candidates: 7362
- Actions without positive candidates: 413
- Actions with original target: 5114
- Actions without original target: 2661
- Actions with candidate geometry: 7350
- Actions without candidate geometry: 425
- Records excluded as invalid: 0

This report contains aggregate metadata only; it does not include page HTML or task text.
