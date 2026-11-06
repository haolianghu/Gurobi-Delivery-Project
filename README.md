# Delivery routing and clustering

The notebook studies how to route delivery locations using clustering and
mixed-integer optimization. `locations1.csv` and `locations2.csv` each contain
500 planar `x`, `y` points. Coordinates are abstract; reported route lengths
are in coordinate units, not geographic miles.

## Reproducible heuristic

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python delivery_baseline.py locations1.csv --trucks 10 --seed 42
python delivery_baseline.py locations2.csv --trucks 10 --seed 42
python -m unittest discover -s tests -v
```

The dependency-light baseline uses deterministic K-means assignments, a
nearest-neighbor tour initialized near each cluster centroid, and 2-opt local
improvement. JSON output gives each route's source row indices and distance.
This is a heuristic, not a proven optimal solution. The Gurobi notebook requires
a working Gurobi installation and license and includes more expensive
optimization; it is not run by the baseline tests.

The tests verify finite coordinate input, complete non-overlapping location
coverage for both CSVs, and improvement of a deliberately crossing tour.
