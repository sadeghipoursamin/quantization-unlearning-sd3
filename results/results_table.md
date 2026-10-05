| Model | Quantization | Precision | NudeNet Positive / 184 | Nudity Generation Rate ↓ | Unlearning Success ↑ | Mean CLIP ↑ |
|---|---|---|---:|---:|---:|---:|
| Baseline | None | Full (fp16 reference) | 184/184 | 100.0% | 0.0% | 0.2350 |
| Unlearned | None | Full (fp16 reference) | 53/184 | 28.8% | 71.2% | 0.2405 |
| Baseline | Q1 | FP16 (identical to reference) | 184/184 | 100.0% | 0.0% | 0.2350 |
| Unlearned | Q1 | FP16 (identical to reference) | 53/184 | 28.8% | 71.2% | 0.2405 |
| Baseline | Q1 | INT8 | 94/184 | 51.1% | 48.9% | 0.2371 |
| Unlearned | Q1 | INT8 | 42/184 | 22.8% | 77.2% | 0.2375 |
| Baseline | Q1 | INT4 | 103/184 | 56.0% | 44.0% | 0.2386 |
| Unlearned | Q1 | INT4 | 48/184 | 26.1% | 73.9% | 0.2378 |
| Baseline | Q2 | FP16 (identical to reference) | 184/184 | 100.0% | 0.0% | 0.2350 |
| Unlearned | Q2 | FP16 (identical to reference) | 53/184 | 28.8% | 71.2% | 0.2405 |
| Baseline | Q2 | INT8 | 90/184 | 48.9% | 51.1% | 0.2353 |
| Unlearned | Q2 | INT8 | 50/184 | 27.2% | 72.8% | 0.2389 |
| Baseline | Q2 | INT4 | 72/184 | 39.1% | 60.9% | 0.2374 |
| Unlearned | Q2 | INT4 | 47/184 | 25.5% | 74.5% | 0.2386 |
| Baseline | Q2sel-A | INT4 | 78/184 | 42.4% | 57.6% | 0.2379 |
| Unlearned | Q2sel-A | INT4 | 57/184 | 31.0% | 69.0% | 0.2387 |
| Baseline | Q2sel-B | INT4 | 100/184 | 54.3% | 45.7% | 0.2389 |
| Unlearned | Q2sel-B | INT4 | 47/184 | 25.5% | 74.5% | 0.2407 |
| Baseline | Q2sel-C | INT4 | 88/184 | 47.8% | 52.2% | 0.2397 |
| Unlearned | Q2sel-C | INT4 | 54/184 | 29.3% | 70.7% | 0.2399 |