import csv
import ast
from collections import defaultdict, Counter

CSV_PATH = "d:/PBL/borg_traces_30000.csv"

def parse_dict(val):
    if not val or val == 'None' or val == '':
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print("Loaded 30,000 rows.")

# Check chronological ordering
times = [float(r['time']) for r in rows if r['time'] != '']
is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
print(f"Is time strictly non-decreasing in original file? {is_sorted}")

# Unique Workload Instances: (collection_id, instance_index)
instance_history = defaultdict(list)
for idx, r in enumerate(rows):
    cid = r['collection_id']
    i_idx = r['instance_index']
    t = float(r['start_time']) if r['start_time'] != '' else float(r['time'])
    key = (cid, i_idx)
    instance_history[key].append((t, idx, r))

print(f"Total Unique Instances (collection_id, instance_index): {len(instance_history)}")

# Sequence lengths per instance
seq_lengths = [len(v) for v in instance_history.values()]
print(f"Max sequence length per instance: {max(seq_lengths)}")
print(f"Min sequence length per instance: {min(seq_lengths)}")
print(f"Avg sequence length per instance: {sum(seq_lengths)/len(seq_lengths):.2f}")

counts = Counter(seq_lengths)
print(f"Distribution of instance sequence lengths: {dict(sorted(counts.items()))}")

# Unique Collections (job/workload group)
coll_history = defaultdict(list)
for idx, r in enumerate(rows):
    cid = r['collection_id']
    t = float(r['start_time']) if r['start_time'] != '' else float(r['time'])
    coll_history[cid].append((t, idx, r))

print(f"\nTotal Unique Collections (collection_id): {len(coll_history)}")
coll_seq_lengths = [len(v) for v in coll_history.values()]
print(f"Max observations per collection: {max(coll_seq_lengths)}")
print(f"Min observations per collection: {min(coll_seq_lengths)}")
print(f"Avg observations per collection: {sum(coll_seq_lengths)/len(coll_seq_lengths):.2f}")

# Machine ID Grouping
mach_history = defaultdict(list)
for idx, r in enumerate(rows):
    mid = r['machine_id']
    t = float(r['start_time']) if r['start_time'] != '' else float(r['time'])
    mach_history[mid].append((t, idx, r))

print(f"\nTotal Unique Machines (machine_id): {len(mach_history)}")
mach_seq_lengths = [len(v) for v in mach_history.values()]
print(f"Max observations per machine: {max(mach_seq_lengths)}")
print(f"Min observations per machine: {min(mach_seq_lengths)}")
print(f"Avg observations per machine: {sum(mach_seq_lengths)/len(mach_seq_lengths):.2f}")

# Check CPU distribution parsing
dist_parsed = 0
for r in rows[:100]:
    raw_dist = r['cpu_usage_distribution']
    if raw_dist and raw_dist != '[]':
        dist_parsed += 1

print(f"\nCPU Usage distribution parsed in first 100 sample: {dist_parsed}/100")
