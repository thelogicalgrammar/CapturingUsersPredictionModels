"""
Aggregate scalability summaries into shareable CSV files.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate scalability summary JSON files."
    )
    parser.add_argument('--input-dir', type=Path, required=True)
    return parser.parse_args()


def serialize_value(value):
    """
    Convert nested values to CSV-safe strings when needed.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_summary_rows(input_dir):
    """
    Load all summary JSON files below one result directory.
    """
    rows = []
    for path in sorted(input_dir.rglob('*_summary.json')):
        with path.open() as file:
            row = json.load(file)
        row['summary_path'] = str(path)
        rows.append(row)
    return rows


def write_rows_csv(rows, output_path):
    """
    Write raw per-run rows to CSV.
    """
    if not rows:
        raise ValueError("No summary rows found to aggregate.")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key)) for key in fieldnames})


def aggregate_rows(rows):
    """
    Aggregate results by model variant and number of covariates.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model_variant'], row['n_covariates'])].append(row)

    aggregated_rows = []
    for (model_variant, n_covariates), group_rows in sorted(grouped.items()):
        runtimes = np.asarray([row['total_runtime'] for row in group_rows], dtype=float)
        divergences = np.asarray([row['n_divergences'] for row in group_rows], dtype=float)
        max_rhats = np.asarray([row['max_rhat'] for row in group_rows], dtype=float)
        min_ess_bulk = np.asarray([row['min_ess_bulk'] for row in group_rows], dtype=float)
        min_ess_tail = np.asarray([row['min_ess_tail'] for row in group_rows], dtype=float)
        aggregated_rows.append({
            'model_variant': model_variant,
            'n_covariates': n_covariates,
            'n_runs': len(group_rows),
            'runtime_mean': float(np.mean(runtimes)),
            'runtime_std': float(np.std(runtimes)),
            'divergences_mean': float(np.mean(divergences)),
            'divergences_max': int(np.max(divergences)),
            'max_rhat_mean': float(np.mean(max_rhats)),
            'max_rhat_max': float(np.max(max_rhats)),
            'min_ess_bulk_mean': float(np.mean(min_ess_bulk)),
            'min_ess_bulk_min': float(np.min(min_ess_bulk)),
            'min_ess_tail_mean': float(np.mean(min_ess_tail)),
            'min_ess_tail_min': float(np.min(min_ess_tail)),
        })
    return aggregated_rows


def main():
    """
    Aggregate raw summary JSON files into raw and grouped CSV tables.
    """
    args = parse_args()
    rows = load_summary_rows(args.input_dir)
    write_rows_csv(rows, args.input_dir / 'all_runs.csv')
    grouped_rows = aggregate_rows(rows)
    write_rows_csv(grouped_rows, args.input_dir / 'grouped_by_variant_and_k.csv')
    print(f"Aggregated {len(rows)} runs from {args.input_dir}")
    print(f"Raw CSV: {args.input_dir / 'all_runs.csv'}")
    print(f"Grouped CSV: {args.input_dir / 'grouped_by_variant_and_k.csv'}")


if __name__ == '__main__':
    main()
