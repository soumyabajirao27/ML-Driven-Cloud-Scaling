import csv
import ast
import os
import numpy as np
from collections import defaultdict

FULL_CSV_PATH = "d:/PBL/borg_traces_data.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

rows = []
with open(FULL_CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, r in enumerate(reader):
        try:
            t = float(r['time'])
            if 0 <= t <= 3.0e12:
                rows.append((t, idx, r))
        except ValueError:
            pass

rows.sort(key=lambda x: x[0])

# Group by instance
instance_groups = defaultdict(list)
for t, idx, r in rows:
    cid = r['collection_id']
    i_idx = r['instance_index']
    if cid and i_idx:
        avg = parse_dict(r['average_usage']) or {}
        req = parse_dict(r['resource_request']) or {}
        cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
        cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
        cpu_util = (cpu_avg / cpu_req * 100.0) if (not np.isnan(cpu_avg) and not np.isnan(cpu_req) and cpu_req > 0) else np.nan
        
        instance_groups[(cid, i_idx)].append({
            'time_us': t,
            'cpu_avg': cpu_avg,
            'cpu_req': cpu_req,
            'cpu_util': cpu_util,
            'row': r
        })

# 1. Audit L=5 sequences extracted with delta_t <= 3600s
l5_gaps = []
l5_target_horizons = []
strict_15m_seqs = 0
strict_30m_seqs = 0
regular_seqs = 0  # max(gap) - min(gap) <= 60s
irregular_seqs = 0

for key, trace in instance_groups.items():
    if len(trace) >= 6:
        trace.sort(key=lambda x: x['time_us'])
        for i in range(len(trace) - 5):
            window = trace[i:i+6]
            is_valid = True
            gaps = []
            for j in range(5):
                dt = (window[j+1]['time_us'] - window[j]['time_us']) / 1e6
                if not (0 < dt <= 3600):
                    is_valid = False
                    break
                gaps.append(dt)
            
            if is_valid:
                l5_gaps.append(gaps)
                l5_target_horizons.append(gaps[-1]) # 5th gap is prediction step
                
                # Check regularity
                gap_spread = max(gaps[:4]) - min(gaps[:4])
                if gap_spread <= 60:
                    regular_seqs += 1
                else:
                    irregular_seqs += 1
                
                if all(g <= 900 for g in gaps):
                    strict_15m_seqs += 1
                if all(g <= 1800 for g in gaps):
                    strict_30m_seqs += 1

total_l5 = len(l5_gaps)
all_internal_gaps = [g for seq in l5_gaps for g in seq[:4]]

print("=" * 80)
print(f"LSTM Sequence Regularity Audit (Total L=5 Sequences: {total_l5:,})")
print("=" * 80)

print(f"Regular Sequences (Spread <= 60s): {regular_seqs} ({regular_seqs/total_l5*100:.2f}%)")
print(f"Irregular Sequences (Spread > 60s): {irregular_seqs} ({irregular_seqs/total_l5*100:.2f}%)")
print(f"Strict Sequences (All gaps <= 15 min): {strict_15m_seqs} ({strict_15m_seqs/total_l5*100:.2f}%)")
print(f"Strict Sequences (All gaps <= 30 min): {strict_30m_seqs} ({strict_30m_seqs/total_l5*100:.2f}%)")

if all_internal_gaps:
    print(f"\nInternal Time Gap Stats (seconds):")
    print(f"  - Min: {min(all_internal_gaps):.1f} s")
    print(f"  - p25: {np.percentile(all_internal_gaps, 25):.1f} s ({(np.percentile(all_internal_gaps, 25)/60):.1f} min)")
    print(f"  - Median: {np.median(all_internal_gaps):.1f} s ({(np.median(all_internal_gaps)/60):.1f} min)")
    print(f"  - Mean: {np.mean(all_internal_gaps):.1f} s ({(np.mean(all_internal_gaps)/60):.1f} min)")
    print(f"  - p75: {np.percentile(all_internal_gaps, 75):.1f} s ({(np.percentile(all_internal_gaps, 75)/60):.1f} min)")
    print(f"  - p90: {np.percentile(all_internal_gaps, 90):.1f} s ({(np.percentile(all_internal_gaps, 90)/60):.1f} min)")
    print(f"  - Max: {max(all_internal_gaps):.1f} s ({(max(all_internal_gaps)/60):.1f} min)")

print(f"\nTarget Horizon Stats (seconds):")
print(f"  - Median: {np.median(l5_target_horizons):.1f} s ({(np.median(l5_target_horizons)/60):.1f} min)")
print(f"  - Mean: {np.mean(l5_target_horizons):.1f} s ({(np.mean(l5_target_horizons)/60):.1f} min)")

# 2. Audit RF/XGBoost Pairs (borg_temporal_pairs.csv)
pairs_csv = "d:/PBL/data/temporal/borg_temporal_pairs.csv"
pair_deltas = []
if os.path.exists(pairs_csv):
    with open(pairs_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                pair_deltas.append(float(r['delta_t_sec']))
            except ValueError:
                pass

print(f"\n--- RF/XGBoost Temporal Pairs Gap Audit ({len(pair_deltas):,} rows) ---")
if pair_deltas:
    print(f"  - Min: {min(pair_deltas):.1f} s")
    print(f"  - p25: {np.percentile(pair_deltas, 25):.1f} s ({(np.percentile(pair_deltas, 25)/60):.1f} min)")
    print(f"  - Median: {np.median(pair_deltas):.1f} s ({(np.median(pair_deltas)/60):.1f} min)")
    print(f"  - Mean: {np.mean(pair_deltas):.1f} s ({(np.mean(pair_deltas)/60):.1f} min)")
    print(f"  - p75: {np.percentile(pair_deltas, 75):.1f} s ({(np.percentile(pair_deltas, 75)/60):.1f} min)")
    print(f"  - Max: {max(pair_deltas):.1f} s ({(max(pair_deltas)/60):.1f} min)")
