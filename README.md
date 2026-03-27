# Graph-Q-SAT
Reproduction of Graph-Q-SAT [NeurIPS'20]

## Evaluation Results

### Median Decision Reduction

| Dataset | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- |
| 50 | 2.36x | 2.38x | 1.87x | 2.00x | 2.29x |
| sr | 1.50x | 1.50x | 1.40x | 1.41x | 1.43x |
| 3-sat | 1.67x | 1.60x | 1.67x | 1.62x | 1.54x |
| ca | 1.43x | 1.38x | 1.29x | 1.34x | 1.37x |
| ps | 1.55x | 1.50x | 1.50x | 1.50x | 1.52x |
| k-clique | 12.00x | 12.00x | 12.33x | 12.00x | 11.55x |
| k-domset | 5.25x | 5.14x | 5.42x | 5.00x | 5.33x |
| k-vercov | 5.80x | 7.27x | 6.90x | 6.50x | 4.89x |

### Median Propagation Reduction

| Dataset | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- |
| 50 | 2.81x | 2.72x | 1.96x | 2.25x | 2.70x |
| sr | 1.15x | 1.20x | 1.19x | 1.22x | 1.22x |
| 3-sat | 1.66x | 1.39x | 1.64x | 1.53x | 1.38x |
| ca | 2.19x | 2.05x | 2.01x | 2.00x | 2.03x |
| ps | 1.30x | 1.33x | 1.28x | 1.30x | 1.32x |
| k-clique | 3.20x | 3.05x | 3.35x | 3.55x | 2.71x |
| k-domset | 5.51x | 5.26x | 5.62x | 5.43x | 5.52x |
| k-vercov | 4.43x | 4.78x | 4.46x | 5.29x | 4.65x |
