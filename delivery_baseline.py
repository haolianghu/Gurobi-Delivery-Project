"""Build deterministic clustered delivery routes from x/y coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_locations(path):
    frame = pd.read_csv(path)
    if not {"x", "y"}.issubset(frame.columns):
        raise ValueError("CSV must contain x and y columns.")
    points = frame[["x", "y"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Coordinates must be nonempty and finite.")
    return points


def route_length(route, distances):
    if len(route) < 2:
        return 0.0
    return float(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


def two_opt(route, distances):
    """Improve a closed route with first-improvement 2-opt edge reversals."""
    route = list(route)
    if len(route) < 4:
        return route
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route) if i else len(route) - 1):
                a, b = route[i], route[(i + 1) % len(route)]
                c, d = route[j], route[(j + 1) % len(route)]
                if distances[a, c] + distances[b, d] + 1e-12 < distances[a, b] + distances[c, d]:
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                    improved = True
                    break
            if improved:
                break
    return route


def plan(points, trucks=10, seed=42):
    if isinstance(trucks, bool) or not isinstance(trucks, int) or not 1 <= trucks <= len(points):
        raise ValueError("Truck count must be between 1 and the number of locations.")
    labels = KMeans(n_clusters=trucks, n_init=10, random_state=seed).fit_predict(points)
    tours = []
    for cluster in range(trucks):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        subset = points[members]
        distances = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=2)
        center = subset.mean(axis=0)
        first = int(np.argmin(np.linalg.norm(subset - center, axis=1)))
        route = [first]
        remaining = set(range(len(members))) - {first}
        while remaining:
            last = route[-1]
            following = min(remaining, key=lambda idx: (distances[last, idx], idx))
            route.append(following)
            remaining.remove(following)
        route = two_opt(route, distances)
        tours.append({
            "truck": cluster,
            "locations": members[route].astype(int).tolist(),
            "distance": route_length(route, distances),
        })
    return {"locations": len(points), "trucks": trucks, "total_distance": sum(t["distance"] for t in tours), "routes": tours}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("locations1.csv"))
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = plan(load_locations(args.csv), args.trucks, args.seed)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
