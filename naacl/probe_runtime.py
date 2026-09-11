#!/usr/bin/env python3
"""Gate 0: compare solo and mixed-load outputs on real observable B1 prefixes.

Run on the allocated compute node against idle deterministic servers. This is a
measurement utility, not a replacement for the full B1/B2/B4 equivalence smoke.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from frontier_common import VLLMClient, json_fingerprint, load_jsonl
from frontier_judge import judge_assistant_response_v5
from run_stage import TARGET, JUDGE, audit


def requests_for(rows):
    candidates=[]
    for r in rows:
        messages=[]
        index=0
        seed=r['rollout_provenance']['record_seed']
        for turn in r['turns']:
            if turn['role']=='assistant':
                candidates.append((sum(len(m['content']) for m in messages),list(messages),
                    seed+1009*(index+1),seed+1000003+1013*(index+1),turn['text']))
                index+=1
            messages.append({'role':turn['role'],'content':turn['text'].strip() if turn['role']=='user' else turn['text']})
    if not candidates: raise RuntimeError('no realized assistant prefixes')
    candidates.sort(key=lambda item:item[0])
    selected=[candidates[i] for i in sorted({0,len(candidates)//3,2*len(candidates)//3,len(candidates)-1})]
    target,judge=[],[]
    class Recorder:
        def chat(self,messages,**kwargs):
            judge.append(dict(messages=messages,kwargs=kwargs))
            return '{"assistant_behavior":"SAFE_HELP","harmfulness":0,"boundary_violation":0,"refusal_strength":0,"confidence":1}'
    for _,prefix,target_seed,judge_seed,response in selected:
        target.append(dict(messages=prefix,kwargs=dict(seed=target_seed,temperature=0.,max_tokens=2048),expected=response))
        judge_assistant_response_v5(Recorder(),prefix+[{'role':'assistant','content':response}],
            seed=judge_seed,max_context_chars=100000)
    return {'target':target,'judge':judge}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Complete deterministic B1 (or B2) JSONL')
    p.add_argument('--target-urls',nargs='*',default=[])
    p.add_argument('--judge-urls',nargs='*',default=[])
    p.add_argument('--concurrency',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not(args.target_urls or args.judge_urls) or min(args.concurrency+[args.repeats])<1:
        p.error('provide endpoints and positive concurrency/repeats')
    rows=load_jsonl(args.input)
    audit('b1',args.input,len(rows))
    workload=requests_for(rows)
    report=[]
    for role,urls in [('target',args.target_urls),('judge',args.judge_urls)]:
        model=TARGET if role=='target' else JUDGE
        cases=workload[role]
        for url in urls:
            client=VLLMClient(model,url)
            if not client.health_check():raise RuntimeError(f'{url} unavailable')
            solo=[]
            for case in cases:
                value=client.chat_result(case['messages'],require_stop=True,require_usage=True,**case['kwargs'])
                if 'expected' in case and value['content']!=case['expected']:
                    raise RuntimeError('GATE 0 FAILED: solo target differs from stored B1')
                solo.append(value)
            for level in args.concurrency:
                start=time.monotonic()
                tokens=0
                for repeat in range(args.repeats):
                    # Start a full wave together, rotating to include every case.
                    jobs=[i%len(cases) for i in range(max(level,len(cases)))]
                    if repeat%2:jobs.reverse()
                    def call(i):
                        case=cases[i]
                        return i,VLLMClient(model,url).chat_result(case['messages'],
                            require_stop=True,require_usage=True,**case['kwargs'])
                    with ThreadPoolExecutor(max_workers=level) as pool:
                        for i,result in pool.map(call,jobs):
                            if result!=solo[i]:
                                raise RuntimeError(f'GATE 0 FAILED: {role} {url} concurrency={level} case={i}')
                            tokens+=result['completion_tokens']
                elapsed=time.monotonic()-start
                row=dict(role=role,server=url,concurrency=level,repeats=args.repeats,
                    seconds=elapsed,completion_tokens=tokens,completion_tokens_per_second=tokens/elapsed,
                    request_fingerprints=[json_fingerprint(c) for c in cases],
                    response_fingerprints=[json_fingerprint(v) for v in solo])
                report.append(row)
                print(json.dumps(row),flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='passed',measurements=report),indent=2)+'\n')
    print('GATE 0 PASSED: exact solo/mixed outputs on the tested prefixes and concurrency levels')


if __name__=='__main__':main()
