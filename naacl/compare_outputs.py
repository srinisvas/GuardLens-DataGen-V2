#!/usr/bin/env python3
"""Require identical complete scientific outputs, including ordering and all fields."""
import argparse
import json
from pathlib import Path
from execution import OutputLease, file_digest
from frontier_common import json_fingerprint
from frontier_common import load_jsonl


def difference(a,b,path='$'):
    if type(a) is not type(b):
        return f'{path}: type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a,dict):
        if a.keys()!=b.keys():
            return f'{path}: missing={sorted(a.keys()-b.keys())} extra={sorted(b.keys()-a.keys())}'
        for key in a:
            diff=difference(a[key],b[key],path+'.'+str(key))
            if diff: return diff
    elif isinstance(a,list):
        if len(a)!=len(b): return f'{path}: lengths {len(a)} != {len(b)}'
        for i,(left,right) in enumerate(zip(a,b)):
            diff=difference(left,right,f'{path}[{i}]')
            if diff: return diff
    elif a!=b:
        # Values may contain training text. Show location, not a prompt dump.
        return f'{path}: values differ'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--optimized',required=True)
    p.add_argument('--trial-id',required=True,help='Expected Slurm job ID or standalone executor trial ID')
    args=p.parse_args()
    with OutputLease(args.optimized):
        compare(args)


def compare(args):
    receipt_path=Path(str(Path(args.optimized).resolve())+'.completion.json')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='complete' or receipt.get('trial_id')!=args.trial_id:
        raise RuntimeError('optimized output is not complete for the requested trial')
    if receipt.get('output')!=str(Path(args.optimized).resolve()) or receipt.get('output_sha256')!=file_digest(args.optimized):
        raise RuntimeError('optimized output differs from its completion receipt')
    contract=receipt.get('contract') or {}
    if not all(k in contract for k in ('stage','input_fingerprint','scientific_config','runtime','code')):
        raise RuntimeError('completion receipt lacks execution provenance')
    if json_fingerprint(contract)!=receipt.get('contract_fingerprint') or contract['input_fingerprint']!=receipt.get('input_fingerprint'):
        raise RuntimeError('completion receipt provenance fingerprint mismatch')
    reference,optimized=load_jsonl(args.reference),load_jsonl(args.optimized)
    if len(optimized)!=receipt.get('record_count'):
        raise RuntimeError('receipt record count mismatch')
    for name,rows in [('reference',reference),('optimized',optimized)]:
        ids=[r.get('conversation_id') for r in rows]
        if not rows or any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
            raise RuntimeError(f'{name}: empty output, invalid IDs or duplicate IDs')
    diff=difference(reference,optimized)
    if diff: raise RuntimeError('SCIENTIFIC OUTPUT MISMATCH: '+diff)
    print(f'EXACT SCIENTIFIC EQUIVALENCE PASSED: {len(reference)} records, every field and record order')


if __name__=='__main__': main()
