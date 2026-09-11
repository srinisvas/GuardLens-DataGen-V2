#!/usr/bin/env python3
"""Summarize measured request timings; do not infer a GPU bottleneck from call counts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir',required=True)
    args=p.parse_args()
    groups=defaultdict(list)
    with open(Path(args.state_dir)/'requests.jsonl') as f:
        for line in f:
            row=json.loads(line)
            if 'elapsed_seconds' in row: groups[(row['model'],row['server'])].append(row)
    for (model,server),rows in sorted(groups.items()):
        elapsed=sorted(r['elapsed_seconds'] for r in rows)
        queue=[r['queue_seconds'] for r in rows]
        tokens=sum((r.get('usage') or {}).get('completion_tokens',0) or 0 for r in rows)
        print(json.dumps(dict(model=model,server=server,requests=len(rows),
            errors=sum('error' in r for r in rows),completion_tokens=tokens,
            median_seconds=statistics.median(elapsed),p95_seconds=elapsed[min(len(elapsed)-1,int(.95*len(elapsed)))],
            mean_client_queue_seconds=statistics.mean(queue)),indent=2))
    print('Latency includes concurrent work and client queueing; it is not additive GPU time.')
    print('Use allocation-*/metrics.jsonl and server logs to inspect waiting requests, GPU utilization, KV pressure and preemptions.')


if __name__=='__main__': main()
