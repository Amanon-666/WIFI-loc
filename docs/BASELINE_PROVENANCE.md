# Baseline provenance

Source: https://github.com/andryr/indoor_localization

Pinned commit: `f34f1bf4560109254887e2f04166dd61e0d9dae9`.

`vendor/andryr/` stores unmodified README and both original notebooks downloaded
from that commit. Code was extracted from notebook cell 9 with Python AST:

- `mlp_longitude_latitude.ipynb`: `class MLP` → `models/andryr.py`.
- `knn.ipynb`: `knn_weight` → `models/wknn.py`.

## Minimal modifications

- MLP: keep six named Linear/BN layers, final head, and forward ordering.
  Constructor widths, BN parameters and LeakyReLU slope now come from YAML.
  V1 values are identical to the original architecture. Initialization changes
  to the agreed V1 Xavier/zero-bias rule. Device selection happens outside the model.
- WKNN: expose the existing `1e-6` epsilon as a config argument; keep inverse-square
  weights and three scan neighbours. Use NumPy Euclidean distances and row-id tie
  breaking because the V1 protocol fixes tie order. No position pre-averaging.
- Replace USERID-based split with the shared manifest, replace global coordinate
  constants with permitted training-position centering, and apply the support-only AP mask.
- MLP source/target update counts and optimizer settings follow V1; do not retain
  notebook validation-based scheduling/early stopping or `.cuda()` assumptions.

`tests/check_invariants.py` executes the original class/weight function from the
saved notebooks. It compares MLP forward outputs under identical weights and
WKNN predictions against the original sklearn estimator on non-tied inputs.
These are implementation equivalence checks, not a claim to reproduce the
notebook's original experimental scores.
