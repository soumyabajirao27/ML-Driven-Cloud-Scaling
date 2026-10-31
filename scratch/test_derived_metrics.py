import csv
import ast
import numpy as np

CSV_PATH = "d:/PBL/borg_traces_30000.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

def parse_float_array(val):
    if not val or val == 'None' or val == '' or val == '[]':
        return []
    try:
        clean = val.strip()
        if clean.startswith('[') and clean.endswith(']'):
            clean = clean[1:-1].strip()
        tokens = clean.replace(',', ' ').split()
        return [float(t) for t in tokens if t != '']
    except Exception:
        return []

processed_records = []
invalid_durations = 0
malformed_records = 0

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        try:
            # Timestamps
            t = float(row['time'])
            st = float(row['start_time'])
            et = float(row['end_time'])
            duration = (et - st) / 1e6  # seconds
            if duration < 0:
                invalid_durations += 1

            # Dicts
            req = parse_dict(row['resource_request']) or {}
            avg = parse_dict(row['average_usage']) or {}
            mx = parse_dict(row['maximum_usage']) or {}
            rnd = parse_dict(row['random_sample_usage']) or {}

            cpu_req = float(req.get('cpus')) if req.get('cpus') is not None else np.nan
            mem_req = float(req.get('memory')) if req.get('memory') is not None else np.nan

            cpu_avg = float(avg.get('cpus')) if avg.get('cpus') is not None else np.nan
            mem_avg = float(avg.get('memory')) if avg.get('memory') is not None else np.nan

            cpu_max = float(mx.get('cpus')) if mx.get('cpus') is not None else np.nan
            mem_max = float(mx.get('memory')) if mx.get('memory') is not None else np.nan

            cpu_rand = float(rnd.get('cpus')) if rnd.get('cpus') is not None else np.nan

            assigned_mem = float(row['assigned_memory']) if row['assigned_memory'] != '' else np.nan
            page_cache_mem = float(row['page_cache_memory']) if row['page_cache_memory'] != '' else np.nan

            cpi = float(row['cycles_per_instruction']) if row['cycles_per_instruction'] != '' else np.nan
            mai = float(row['memory_accesses_per_instruction']) if row['memory_accesses_per_instruction'] != '' else np.nan

            # Usage distributions
            cpu_dist = parse_float_array(row['cpu_usage_distribution'])
            tail_dist = parse_float_array(row['tail_cpu_usage_distribution'])

            cpu_dist_median = np.median(cpu_dist) if len(cpu_dist) > 0 else np.nan
            cpu_dist_max = np.max(cpu_dist) if len(cpu_dist) > 0 else np.nan
            tail_dist_max = np.max(tail_dist) if len(tail_dist) > 0 else np.nan

            # Derived metrics
            cpu_util_pct = (cpu_avg / cpu_req * 100.0) if (not np.isnan(cpu_avg) and not np.isnan(cpu_req) and cpu_req > 0) else np.nan
            mem_util_pct = (mem_avg / mem_req * 100.0) if (not np.isnan(mem_avg) and not np.isnan(mem_req) and mem_req > 0) else np.nan
            cpu_burst = (cpu_max / cpu_avg) if (not np.isnan(cpu_max) and not np.isnan(cpu_avg) and cpu_avg > 0) else np.nan

            processed_records.append({
                'time_sec': t / 1e6,
                'duration_sec': duration,
                'cpu_req': cpu_req,
                'mem_req': mem_req,
                'cpu_avg': cpu_avg,
                'mem_avg': mem_avg,
                'cpu_max': cpu_max,
                'mem_max': mem_max,
                'cpu_rand': cpu_rand,
                'assigned_mem': assigned_mem,
                'page_cache_mem': page_cache_mem,
                'cpi': cpi,
                'mai': mai,
                'cpu_dist_median': cpu_dist_median,
                'cpu_dist_max': cpu_dist_max,
                'tail_dist_max': tail_dist_max,
                'cpu_util_pct': cpu_util_pct,
                'mem_util_pct': mem_util_pct,
                'cpu_burst': cpu_burst,
                'priority': float(row['priority']) if row['priority'] != '' else np.nan,
                'scheduling_class': float(row['scheduling_class']) if row['scheduling_class'] != '' else np.nan,
                'failed': int(row['failed']) if row['failed'] != '' else 0
            })
        except Exception as e:
            malformed_records += 1

print(f"Successfully processed records: {len(processed_records)}")
print(f"Invalid durations: {invalid_durations}")
print(f"Malformed records: {malformed_records}")

# Check metric distributions
cpu_utils = [r['cpu_util_pct'] for r in processed_records if not np.isnan(r['cpu_util_pct'])]
mem_utils = [r['mem_util_pct'] for r in processed_records if not np.isnan(r['mem_util_pct'])]
bursts = [r['cpu_burst'] for r in processed_records if not np.isnan(r['cpu_burst'])]

print(f"\n--- Derived Metrics Stats ---")
print(f"CPU Util Pct: count={len(cpu_utils)}, min={np.min(cpu_utils):.2f}%, max={np.max(cpu_utils):.2f}%, median={np.median(cpu_utils):.2f}%, mean={np.mean(cpu_utils):.2f}%")
print(f"Mem Util Pct: count={len(mem_utils)}, min={np.min(mem_utils):.2f}%, max={np.max(mem_utils):.2f}%, median={np.median(mem_utils):.2f}%, mean={np.mean(mem_utils):.2f}%")
print(f"CPU Burst Factor: count={len(bursts)}, min={np.min(bursts):.2f}, max={np.max(bursts):.2f}, median={np.median(bursts):.2f}, mean={np.mean(bursts):.2f}")
