# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin) — AI CUP 2026 / TEAM_10049
"""Realigned test submission — tune to the REAL (N/A-excluded) metric, optionally drop rules/prior.

vs predict_test_submission.py the only changes are:
  * model_weights & score_val use f1_score(labels=SCORED[f]) — timeline 4-class, quality 3-class
    (N/A EXCLUDED, matching the LB). evidence stays 3-class incl N/A; promise 2-class.
  * --no_rules / --no_prior / --no_bestsrc flags (honest 5-fold: pure-BERT beats rules+prior +0.0168 real).
Defaults = pure BERT (no rules, no prior, no best_src override), real-metric tuned.
Runs locally: val preds /tmp/val_io, test preds /tmp/test_io, spans data_set/claude_spans.
"""
import argparse, json, re, numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar

FIELDS = ['promise_status','verification_timeline','evidence_status','evidence_quality']
LABELS = {'promise_status':['Yes','No'],
          'verification_timeline':['already','within_2_years','between_2_and_5_years','more_than_5_years','N/A'],
          'evidence_status':['Yes','No','N/A'],
          'evidence_quality':['Clear','Not Clear','Misleading','N/A']}
# REAL LB metric: timeline & quality EXCLUDE N/A; evidence includes N/A; promise 2-class
SCORED = {'promise_status':['Yes','No'],
          'verification_timeline':['already','within_2_years','between_2_and_5_years','more_than_5_years'],
          'evidence_status':['Yes','No','N/A'],
          'evidence_quality':['Clear','Not Clear','Misleading']}
W = {'promise_status':0.20,'evidence_status':0.30,'verification_timeline':0.15,'evidence_quality':0.35}
BERT_NAMES = ['v1_robertawwm','v2_xlmr_large','v3_bgem3','v5_bgem3_aug','v6_bgem3_augfull',
              'v7_bgem3_aug_all','v8_ckip','v13_phase1','v18_timetok_t4','v19_robertalarge',
              'v20_macbert_rdrop_msd','v21_bgem3_pool_v5','v37_bgem3_dualspans','v40_bgem3_joint_span_w3']
REFERENCE_YEAR = 2024
PAST_DONE_KW=['已建立','已導入','已取得','已通過','已完成','已成立','已認證','已實施','已執行','已推動','已開始','已達成','已簽署','已加入','已發行','已發布','已上線','已揭露','已參與','已制訂','已制定','已實行','已採用','已修訂','已新增','已調整','已優化']
OPERATIONAL_KW=['每年','年度','定期','常態','常設','例行','逐年','逐月','每季','每月','不定期','即時','常年','長期維持','本年度','本期','當期']
PAST_VERB_KW=['導入','建立','取得','通過','完成','成立','推出','獲得','簽署','發行','發布','上線','揭露','參與','制訂','制定','修訂','採用']
LONG_TERM_KW=['長期','中長期','未來 5 年','未來五年','未來十年','永續經營','永續發展願景','長遠','持續推動長期']
NET_ZERO_KW=['Net Zero','net zero','淨零','SBTi','Science Based Targets','科學基礎','碳中和','溫室氣體淨零']
NOT_CLEAR_KW=['積極','持續','努力','加強','提升','優化','強化']

def predict_timeline_rule(r):
    if r.get('promise_status')=='No': return 'N/A'
    ps=r.get('promise_string','') or ''; es=r.get('evidence_string','') or ''; text=ps+' '+es
    if not (ps or es): text=r.get('data','')
    years=[int(y) for y in re.findall(r'\b(20\d{2})\b',text)]
    fy=[y for y in years if y>REFERENCE_YEAR]; py=[y for y in years if y<=REFERENCE_YEAR]
    if fy:
        d=max(fy)-REFERENCE_YEAR
        return 'within_2_years' if d<=2 else 'between_2_and_5_years' if d<=5 else 'more_than_5_years'
    if any(k in text for k in NET_ZERO_KW) and not any(k in text for k in PAST_DONE_KW): return 'more_than_5_years'
    if any(k in text for k in LONG_TERM_KW) and not any(k in text for k in PAST_DONE_KW): return 'more_than_5_years'
    if any(k in text for k in PAST_DONE_KW): return 'already'
    if any(k in text for k in OPERATIONAL_KW): return 'already'
    if py and any(k in text for k in PAST_VERB_KW): return 'already'
    return 'between_2_and_5_years'
def has_specific_numbers(t): return bool(re.search(r'\d+\.\d+|\d{3,}|\d+\s*%|\d+\s*％',t))
def predict_quality_rule(r):
    if r.get('promise_status')=='No': return 'N/A'
    if r.get('evidence_status') in ('No','N/A'): return 'N/A'
    es=r.get('evidence_string','') or ''
    if not es: return None
    if '｜' in es:
        segs=[s.strip() for s in es.split('｜') if s.strip()]
        if any(len(s)<10 for s in segs): return 'Not Clear'
    if has_specific_numbers(es): return 'Clear'
    if len(es)<30: return 'Not Clear'
    if sum(1 for k in NOT_CLEAR_KW if k in es)>=2: return 'Not Clear'
    return None
def predict_evidence_rule(r):
    if r.get('promise_status')=='No': return 'N/A'
    if has_specific_numbers(r.get('data','')): return 'Yes'
    return None
def main_type(t):
    t=(t or '').strip(); return t[0] if t and t[0] in 'ESG' else 'OTHER'
def softmax_logp(probs,T):
    log=np.log(np.maximum(probs,1e-12)); s=log/T; s=s-s.max(axis=-1,keepdims=True); e=np.exp(s); return e/e.sum(axis=-1,keepdims=True)
def fit_T(pl,yi):
    def nll(T): return sum(-np.log(max(softmax_logp(np.array(p),T)[g],1e-12)) for p,g in zip(pl,yi))
    return minimize_scalar(nll,bounds=(0.1,10),method='bounded').x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--val',default='data_set/vpesg4k_val_1000.json')
    ap.add_argument('--val_pred_dir',default='/tmp/val_io')
    ap.add_argument('--test',default='data_set/vpesg4k_test_2000.json')
    ap.add_argument('--test_pred_dir',default='/tmp/test_io')
    ap.add_argument('--val_spans',default='data_set/claude_spans/val_spans_raw.json')
    ap.add_argument('--test_spans',default='data_set/claude_spans/test_spans_raw.json')
    ap.add_argument('--priors',default='pipeline/evaluation/rules/esg_type_priors.json')
    ap.add_argument('--out',required=True)
    ap.add_argument('--rules',action='store_true'); ap.add_argument('--prior',action='store_true')
    ap.add_argument('--bestsrc',action='store_true')
    ap.add_argument('--metric',choices=['real','incl'],default='real')
    ap.add_argument('--gate_votes_val',default=None,help='LLM val votes json (for gate validation)')
    ap.add_argument('--gate_votes_test',default=None,help='LLM test votes json (uncertain subset)')
    ap.add_argument('--gate_tau',type=float,default=0.1,help='override evidence with LLM where evidence margin<tau')
    args=ap.parse_args()
    gate_val={str(x['id']):x for x in json.load(open(args.gate_votes_val))} if args.gate_votes_val else {}
    gate_test={str(x['id']):x for x in json.load(open(args.gate_votes_test))} if args.gate_votes_test else {}
    LAB=lambda f:(SCORED[f] if args.metric=='real' else LABELS[f])

    val=json.load(open(args.val)); test=json.load(open(args.test))
    val_by={str(v['id']):v for v in val}; test_by={str(t['id']):t for t in test}
    priors=json.load(open(args.priors))
    val_spans={x['id']:x for x in json.load(open(args.val_spans))}
    test_spans={x['id']:x for x in json.load(open(args.test_spans))}
    def span_view(r,spans):
        rr=dict(r); cs=spans.get(str(r['id']),{})
        rr['promise_string']=(cs.get('promise_string') or '').strip(); rr['evidence_string']=(cs.get('evidence_string') or '').strip(); return rr
    def load(base,fname):
        return {n:{str(p['id']):p for p in json.load(open(f"{base}/bert_{n}/{fname}"))} for n in BERT_NAMES}
    val_preds=load(args.val_pred_dir,'predictions_with_conf.json')
    test_preds=load(args.test_pred_dir,'predictions_test.json')
    val_ids=[str(v['id']) for v in val if all(str(v['id']) in d for d in val_preds.values())]
    test_ids=[str(t['id']) for t in test]

    calib_val,calib_test={},{}
    for n in BERT_NAMES:
        calib_val[n],calib_test[n]={},{}
        for f in FIELDS:
            T=fit_T([val_preds[n][s]['probs_'+f] for s in val_ids],[LABELS[f].index(val_by[s][f]) for s in val_ids])
            calib_val[n][f]={s:softmax_logp(np.array(val_preds[n][s]['probs_'+f]),T).tolist() for s in val_ids}
            calib_test[n][f]={s:softmax_logp(np.array(test_preds[n][s]['probs_'+f]),T).tolist() for s in test_ids if s in test_preds[n]}
    model_weights={f:{} for f in FIELDS}
    for f in FIELDS:
        y=[val_by[s][f] for s in val_ids]
        for n in BERT_NAMES:
            p=[LABELS[f][int(np.argmax(calib_val[n][f][s]))] for s in val_ids]
            model_weights[f][n]=f1_score(y,p,labels=LAB(f),average='macro',zero_division=0)
    best_src={f:max(model_weights[f],key=lambda n:model_weights[f][n]) for f in FIELDS}
    rules_val={s:{'T':predict_timeline_rule(span_view(val_by[s],val_spans)),'Q':predict_quality_rule(span_view(val_by[s],val_spans)),'E':predict_evidence_rule(val_by[s])} for s in val_ids}
    rules_test={s:{'T':predict_timeline_rule(span_view(test_by[s],test_spans)),'Q':predict_quality_rule(span_view(test_by[s],test_spans)),'E':predict_evidence_rule(test_by[s])} for s in test_ids}
    prior_ratio={}
    for f in FIELDS:
        prior_ratio[f]={}; marg=priors[f]['_marginal']
        for t in ['E','S','G']: prior_ratio[f][t]=np.array([priors[f][t][l]/marg[l] for l in LABELS[f]])
        prior_ratio[f]['OTHER']=np.ones(len(LABELS[f]))
    PRIOR_FIELDS={'verification_timeline','evidence_quality'}

    def assemble(sid,calib,rules,stype,t,offsets,gate=None):
        picks={}
        for f in FIELDS:
            probs=np.zeros(len(LABELS[f]))
            for n in BERT_NAMES:
                if sid in calib[n][f]: probs+=model_weights[f][n]*np.array(calib[n][f][sid])
            ru=rules[sid]
            if args.rules and f=='verification_timeline' and ru['T'] in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru['T'])]=1;probs+=3*oh
            if args.rules and f=='evidence_quality' and ru['Q'] and ru['Q'] in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru['Q'])]=1;probs+=3*oh
            if args.rules and f=='evidence_status' and ru['E'] and ru['E'] in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru['E'])]=1;probs+=3*oh
            if args.prior and f in PRIOR_FIELDS: probs=probs*np.power(prior_ratio[f].get(stype,prior_ratio[f]['OTHER']),0.5)
            if offsets and f in offsets: probs=probs/(probs.sum()+1e-12)+np.array(offsets[f])
            if args.bestsrc:
                bc=max(calib[best_src[f]][f][sid]) if sid in calib[best_src[f]][f] else 0
                bp=LABELS[f][int(np.argmax(calib[best_src[f]][f][sid]))] if sid in calib[best_src[f]][f] else None
                picks[f]=bp if (bp and bc>=t[f]) else LABELS[f][int(probs.argmax())]
            else:
                picks[f]=LABELS[f][int(probs.argmax())]
            # LLM evidence GATE: on BERT-uncertain evidence (margin<tau), trust the LLM diversity vote
            if gate is not None and f=='evidence_status':
                pn=probs/(probs.sum()+1e-12); sr=np.sort(pn)[::-1]
                if (sr[0]-sr[1])<args.gate_tau:
                    vote=gate.get(sid,{}).get('evidence_status')
                    if vote in SCORED[f] and vote!=picks[f]: picks[f]=vote
        if picks['promise_status']=='No': picks['verification_timeline']=picks['evidence_status']=picks['evidence_quality']='N/A'
        elif picks['evidence_status'] in ('No','N/A'): picks['evidence_quality']='N/A'
        return picks
    def score_val(t,offsets,per=False):
        d={}
        for f in FIELDS:
            y=[val_by[s][f] for s in val_ids]
            p=[assemble(s,calib_val,rules_val,main_type(val_by[s].get('esg_type')),t,offsets)[f] for s in val_ids]
            d[f]=f1_score(y,p,labels=LAB(f),average='macro',zero_division=0)
        tot=sum(d[f]*W[f] for f in FIELDS)
        return (tot,d) if per else tot
    bt={f:0 for f in FIELDS}
    if args.bestsrc:
        for _ in range(2):
            for f in FIELDS:
                bw,bv=-1,bt[f]
                for tt in np.linspace(0,1,21):
                    w=score_val({**bt,f:tt},None)
                    if w>bw: bw,bv=w,tt
                bt[f]=bv
    grid=[-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.35,0.5]
    offsets={f:[0.0]*len(LABELS[f]) for f in FIELDS}
    for _ in range(2):
        for f in FIELDS:
            for ci in range(len(LABELS[f])):
                bw,bv=-1,offsets[f][ci]
                for v in grid:
                    offsets[f][ci]=v; w=score_val(bt,offsets)
                    if w>bw: bw,bv=w,v
                offsets[f][ci]=bv
    tot,d=score_val(bt,offsets,per=True)
    print(f">> cfg rules={args.rules} prior={args.prior} bestsrc={args.bestsrc} metric={args.metric}")
    print(f">> val(dev) {args.metric}-metric WF1={tot:.4f}  P={d['promise_status']:.3f} E={d['evidence_status']:.3f} T={d['verification_timeline']:.3f} Q={d['evidence_quality']:.3f}")
    # --- LLM evidence GATE: validate on val (against THIS offset-tuned base), then apply on test ---
    if gate_val:
        def score_gate(gate):
            dd={}
            for f in FIELDS:
                y=[val_by[s][f] for s in val_ids]
                p=[assemble(s,calib_val,rules_val,main_type(val_by[s].get('esg_type')),bt,offsets,gate)[f] for s in val_ids]
                dd[f]=f1_score(y,p,labels=LAB(f),average='macro',zero_division=0)
            return sum(dd[f]*W[f] for f in FIELDS),dd
        g,gd=score_gate(gate_val)
        print(f">> GATE val WF1={g:.4f} ({g-tot:+.4f})  E={gd['evidence_status']:.3f} (base {d['evidence_status']:.3f})  [tau={args.gate_tau}]")
    out=[]
    for tr in test:
        sid=str(tr['id'])
        picks=assemble(sid,calib_test,rules_test,main_type(tr.get('esg_type')),bt,offsets,gate_test or None)
        out.append({'id':tr['id'],**picks,
                    'promise_string':(test_spans.get(sid,{}).get('promise_string') or ''),
                    'evidence_string':(test_spans.get(sid,{}).get('evidence_string') or '')})
    json.dump(out,open(args.out,'w'),ensure_ascii=False,indent=2)
    print(f"✅ {len(out)} → {args.out}")

if __name__=='__main__': main()
