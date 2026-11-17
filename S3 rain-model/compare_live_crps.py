"""Recompute the screenshot's live CRPS from an actual organizer export.

Usage: .venv/bin/python "S3 rain-model/compare_live_crps.py" export.json

Required JSON: submission_id, issue_time, cities (45 unique names in order),
horizons [6, 48], members (45, 2, M), observations (45, 2).
No predictions or observations are reconstructed or inferred by this script.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pairwise_crps(members, observations):
    """Direct definition, independent of the notebook's sorted formula."""
    x = np.maximum(np.asarray(members, dtype=float), 0)
    y = np.asarray(observations, dtype=float)
    return np.abs(x - y[..., None]).mean(axis=-1) - 0.5 * np.abs(
        x[..., :, None] - x[..., None, :]
    ).mean(axis=(-2, -1))


def compare_export(path):
    data = json.loads(Path(path).read_text())
    required = {'submission_id', 'issue_time', 'cities', 'horizons',
                'members', 'observations'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Missing raw export fields: {sorted(missing)}')
    if data['submission_id'] != 9439:
        raise ValueError('This comparison targets submission 9439.')
    if data['issue_time'] != '2026-10-19T11:00:00Z':
        raise ValueError('The screenshot forecast was issued 2026-10-19T11:00:00Z.')
    if data['horizons'] != [6, 48]:
        raise ValueError('Expected horizons in order [6, 48].')
    cities = data['cities']
    if len(cities) != 45 or len(set(cities)) != 45:
        raise ValueError('Expected 45 unique city names in the forecast order.')
    x = np.asarray(data['members'], dtype=float)
    y = np.asarray(data['observations'], dtype=float)
    if x.ndim != 3 or x.shape[:2] != (45, 2) or not 1 <= x.shape[2] <= 100:
        raise ValueError(f'Invalid member shape {x.shape}; expected (45, 2, M).')
    if y.shape != (45, 2):
        raise ValueError(f'Invalid observation shape {y.shape}; expected (45, 2).')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Finite members and finite nonnegative observations required.')

    losses = pairwise_crps(x, y)
    by_horizon = losses.mean(axis=0)
    computed = np.r_[by_horizon, by_horizon.mean()]
    reported = np.array([0.22032332000000002, 0.1774678288888889,
                         0.19889557444444445])
    labels = ['crps_6h', 'crps_48h', 'mean_crps']
    # Allow rounding in exported/platform per-city scores, while recording
    # every numerical difference explicitly.
    matches = bool(np.allclose(computed, reported, rtol=0, atol=5e-8))
    summary = {
        'submission_id': data['submission_id'],
        'issue_time': data['issue_time'],
        'source_export': str(Path(path).resolve()),
        'comparison': {
            name: {'computed': float(a), 'challenge': float(b),
                   'difference': float(a-b)}
            for name, a, b in zip(labels, computed, reported)
        },
        'matches_challenge_within_5e-8': matches,
    }
    out = Path(path).resolve().parent
    with (out / 'live_crps_per_city.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['city', 'horizon_hours', 'observed_mm', 'crps_mm'])
        for j, city in enumerate(cities):
            for k, horizon in enumerate(data['horizons']):
                writer.writerow([city, horizon, y[j, k], losses[j, k]])
    (out / 'live_crps_comparison.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', help='Actual raw forecast and observation JSON export')
    args = parser.parse_args()
    result = compare_export(args.export)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['matches_challenge_within_5e-8'] else 1)
