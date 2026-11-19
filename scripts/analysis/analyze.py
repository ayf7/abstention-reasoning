import json, os, sys, math
from collections import defaultdict

ROOT='/home/ayf7/repos/abstention-reasoning/artifacts/competition_math/hint/models/rl'
ARMS={'p0.1':'qwen2.5-3b-math-hint-p0.1','p0.05':'qwen2.5-3b-math-hint-p0.05',
      'p0.025':'qwen2.5-3b-math-hint-p0.025','p0':'qwen2.5-3b-math-hint-p0'}

def classify_ending(out):
    s=out.rstrip()
    if s.endswith('</answer>'): return 'answer'
    if s.endswith('<abstain>'): return 'abstain'
    if s.endswith('<request></request>'): return 'cut_at_request'
    if '</response>' in s[-40:]: return 'cut_after_hint'
    return 'cut_other'

def analyze_step(path):
    rows=[json.loads(l) for l in open(path)]
    stats=defaultdict(float); n=len(rows)
    # recount hints from text
    for r in rows:
        r['h']=r['output'].count('<response>')  # includes "No more hints available"
        r['req']=r['output'].count('<request></request>')
        r['end']=classify_ending(r['output'])
    stats['n']=n
    stats['hint_rate']=sum(r['h']>0 for r in rows)/n
    stats['req_rate']=sum(r['req']>0 for r in rows)/n
    stats['pct0']=sum(r['h']==0 for r in rows)/n
    stats['pct1']=sum(r['h']==1 for r in rows)/n
    stats['acc']=sum(r['correct'] for r in rows)/n
    stats['malf']=sum(r['malformed'] for r in rows)/n
    stats['score_mean']=sum(r['score'] for r in rows)/n
    stats['len_chars']=sum(len(r['output']) for r in rows)/n
    hint=[r for r in rows if r['h']>0]; noh=[r for r in rows if r['h']==0]
    def agg(rs,key):
        return sum(key(r) for r in rs)/len(rs) if rs else float('nan')
    stats['n_hint']=len(hint)
    stats['acc_hint']=agg(hint,lambda r:r['correct'])
    stats['acc_nohint']=agg(noh,lambda r:r['correct'])
    stats['malf_hint']=agg(hint,lambda r:r['malformed'])
    stats['malf_nohint']=agg(noh,lambda r:r['malformed'])
    stats['len_hint']=agg(hint,lambda r:len(r['output']))
    stats['len_nohint']=agg(noh,lambda r:len(r['output']))
    stats['score_hint']=agg(hint,lambda r:r['score'])
    stats['score_nohint']=agg(noh,lambda r:r['score'])
    stats['basescore_hint']=agg(hint,lambda r:r['score_wo_hint_penalty'])
    stats['basescore_nohint']=agg(noh,lambda r:r['score_wo_hint_penalty'])
    # ending distribution for hint rollouts
    for e in ['answer','abstain','cut_at_request','cut_after_hint','cut_other']:
        stats['end_%s_hint'%e]=agg(hint,lambda r:r['end']==e)
        stats['end_%s_all'%e]=sum(r['end']==e for r in rows)/n
    # within-group advantage of hint rollouts (GRPO normalized)
    advs=[]; adv_base=[]; mixed_diff_correct=[]; mixed_diff_base=[]
    for g in range(0,n,8):
        grp=rows[g:g+8]
        scores=[r['score'] for r in grp]
        m=sum(scores)/8
        var=sum((s-m)**2 for s in scores)/7
        sd=math.sqrt(var)
        bs=[r['score_wo_hint_penalty'] for r in grp]
        mb=sum(bs)/8
        vb=sum((s-mb)**2 for s in bs)/7; sdb=math.sqrt(vb)
        gh=[r for r in grp if r['h']>0]; gn=[r for r in grp if r['h']==0]
        for r in gh:
            advs.append((r['score']-m)/(sd+1e-6))
            adv_base.append((r['score_wo_hint_penalty']-mb)/(sdb+1e-6))
        if gh and gn:
            mixed_diff_correct.append(sum(r['correct'] for r in gh)/len(gh)-sum(r['correct'] for r in gn)/len(gn))
            mixed_diff_base.append(sum(r['score_wo_hint_penalty'] for r in gh)/len(gh)-sum(r['score_wo_hint_penalty'] for r in gn)/len(gn))
    stats['adv_hint_mean']=sum(advs)/len(advs) if advs else float('nan')
    stats['adv_hint_n']=len(advs)
    stats['advbase_hint_mean']=sum(adv_base)/len(adv_base) if adv_base else float('nan')
    stats['mixedgrp_dcorrect']=sum(mixed_diff_correct)/len(mixed_diff_correct) if mixed_diff_correct else float('nan')
    stats['mixedgrp_dbase']=sum(mixed_diff_base)/len(mixed_diff_base) if mixed_diff_base else float('nan')
    stats['n_mixed_groups']=len(mixed_diff_correct)
    return stats

def main():
    steps=sorted(set(list(range(1,61,3))+list(range(70,401,10))+list(range(180,281,5))))
    out={}
    for arm,d in ARMS.items():
        rdir=os.path.join(ROOT,d,'rollouts')
        arm_out={}
        for s in steps:
            p=os.path.join(rdir,'%d.jsonl'%s)
            if os.path.exists(p):
                arm_out[s]=analyze_step(p)
        out[arm]=arm_out
        print('done',arm,file=sys.stderr)
    json.dump(out,open('/tmp/claude-1640984/-home-ayf7-repos-abstention-reasoning/3ec4c588-8eda-44b6-a0c6-e8c8485dfb19/scratchpad/stats.json','w'))

main()
