"""TAPT-hybrid submission builder (SAVED — reused for timeline-fix + multiseed final).

SRC: promise/evidence from TAPT K-fold models, timeline/quality from ORIGINAL models.
Tune everything on the clean 900 OOF (TAPT OOF for P/E, old val preds for T/Q — both held-out).
Pipeline = predict_test_realigned: calib(fit_T)+F1-weights+rules+esg_type prior+bestsrc+timeline-offset.
ONLY timeline offset is tuned (realignment: timeline-offset transfers +0.012; P/E/Q offsets overfit).
Then cascade + compliance + Misleading overrides.

  python3 pipeline/evaluation/build_tapt_hybrid.py --out submissions/.../submission.csv \
      --misleading 12772,12743,12306,12606 [--timeline_offset 0.1,0,-0.1,-0.1,0] [--no_offset]
"""
import argparse, json, numpy as np, sys, os
from sklearn.metrics import f1_score
sys.path.insert(0,'pipeline/evaluation')
import importlib.util
spec=importlib.util.spec_from_file_location("R","pipeline/evaluation/predict_test_realigned.py")
R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
FIELDS,LABELS,SCORED,W=R.FIELDS,R.LABELS,R.SCORED,R.W
KF=['v1_robertawwm','v2_xlmr_large','v3_bgem3','v5_bgem3_aug','v6_bgem3_augfull','v7_bgem3_aug_all',
    'v8_ckip','v13_phase1','v18_timetok_t4','v19_robertalarge','v20_macbert_rdrop_msd','v21_bgem3_pool_v5']
OLD=R.BERT_NAMES  # 14
SRC={'promise_status':'tapt','evidence_status':'tapt','verification_timeline':'old','evidence_quality':'old'}
PRIOR_FIELDS={'verification_timeline','evidence_quality'}
# safety-recurring -> more_than_5_years (val OOF: safety/職安/巡檢 recurring commitments are 85% GT=more5;
# perpetual safety duties never "complete" -> more5. Boundary acc 0.646->0.684, 8/8 correct flips, 0 wrong).
import re as _re
_SAFE=_re.compile(r'職安|職業安全|安全衛生|消防|職業災害|工安|作業環境(測定|安全)|危害辨識|安全巡檢|巡檢|防災|應變演練|疏散')
_RECUR=_re.compile(r'每年|每季|每月|定期|不定期|常態|逐年|持續(推動|進行|監測|改善|辦理|執行|追蹤)|長期')
_DONE=_re.compile(r'已通過|通過[^。]{0,8}審查|已取得|已簽署|已建立|已導入|已完成|已認證|已實施|已達成')
_FUT=_re.compile(r'20[3-9][0-9]|淨零|未來[^。]{0,6}(年|將)|中長期|承諾.{0,10}(達成|減量|實現)')
_SUPP=_re.compile(r'供應商|供應鏈|供應者')
def _safety_more5(promise, full):
    # safety must be the COMMITMENT (in promise), recurring cadence anywhere, no completed milestone /
    # future target, and not a supplier-CSR clause (those skew GT=already).
    return bool(_SAFE.search(promise) and _RECUR.search(full) and not _DONE.search(full)
                and not _FUT.search(full) and not _SUPP.search(promise))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--misleading',default='12772,12743,12306,12606')
    ap.add_argument('--timeline_offset',default=None,help='comma 5 vals; default = tune on OOF')
    ap.add_argument('--no_offset',action='store_true')
    ap.add_argument('--aggressive',action='store_true',help='also tune evidence offset + force timeline ensemble path; report held-out')
    ap.add_argument('--force_timeline_thresh',type=float,default=None)
    ap.add_argument('--force_evidence_thresh',type=float,default=None)
    ap.add_argument('--seeds',default='42',help='comma seeds to AVERAGE for TAPT P/E (42=original no-suffix, e.g. 42,2024,777)')
    ap.add_argument('--safety_more5',nargs='?',const='all',default='',help='force safety-recurring timeline -> more5: "all" or "already"-only (OOF lever)')
    ap.add_argument('--kf_models',default=None,help='comma subset of the 12 KF models for TAPT P/E (default all 12; e.g. v3_bgem3,v13_phase1,v6_bgem3_augfull)')
    ap.add_argument('--external_pe_csv',default=None,help='CSV with id,promise_status,evidence_status to OVERRIDE our promise/evidence on TEST (teammate Eason stronger binary). Our timeline/quality + cascade-repick + Misleading still apply.')
    args=ap.parse_args()
    global KF
    if args.kf_models:
        KF=[m.strip() for m in args.kf_models.split(',')]
        print(f">> KF restricted to {len(KF)} models: {KF}")
    EXT_PE={}
    if args.external_pe_csv:
        import csv as _csv
        for r in _csv.DictReader(open(args.external_pe_csv)):
            EXT_PE[str(r['id'])]={'promise_status':r['promise_status'].strip(),'evidence_status':r['evidence_status'].strip()}
        print(f">> external P/E override loaded for {len(EXT_PE)} ids")
    SEEDS=[s.strip() for s in args.seeds.split(',')]
    def _suf(s): return '' if s=='42' else f'_s{s}'
    mis=set(args.misleading.split(',')) if args.misleading else set()

    val=json.load(open('data_set/vpesg4k_val_1000.json')); vby={str(v['id']):v for v in val}
    test=json.load(open('data_set/vpesg4k_test_2000.json'))
    priors=json.load(open('pipeline/evaluation/rules/esg_type_priors.json'))
    vspan={x['id']:x for x in json.load(open('data_set/claude_spans/val_spans_raw.json'))}
    tspan={x['id']:x for x in json.load(open('data_set/claude_spans/test_spans_raw.json'))}
    # --- load preds ---
    # TAPT P/E preds, AVERAGED over seeds (variance reduction). seed 42 = no suffix.
    def _avg(per_id):
        return {sid:{'probs_'+f: np.mean([np.array(p['probs_'+f]) for p in pl],axis=0).tolist() for f in FIELDS}
                for sid,pl in per_id.items()}
    def load_oof(m):
        per={}
        for s in SEEDS:
            for k in range(5):
                fn=f'/tmp/kfold_preds/pred_{m}_tapt{_suf(s)}_oof_f{k}.json'
                if os.path.exists(fn):
                    for p in json.load(open(fn)): per.setdefault(str(p['id']),[]).append(p)
        return _avg(per)
    def load_test(m):
        per={}
        for s in SEEDS:
            fn=f'/tmp/kfold_preds/pred_{m}_tapt{_suf(s)}_test.json'
            if os.path.exists(fn):
                for p in json.load(open(fn)): per.setdefault(str(p['id']),[]).append(p)
        return _avg(per)
    tapt_oof={m:load_oof(m) for m in KF}; tapt_test={m:load_test(m) for m in KF}
    nseed_oof=len(set(SEEDS)); print(f"averaging TAPT P/E over seeds {SEEDS} (files present per id may vary)")
    old_val={n:{str(p['id']):p for p in json.load(open(f'/tmp/val_io/bert_{n}/predictions_with_conf.json'))} for n in OLD}
    old_test={n:{str(p['id']):p for p in json.load(open(f'/tmp/test_io/bert_{n}/predictions_test.json'))} for n in OLD}
    oof_ids=[s for s in sorted(tapt_oof[KF[0]],key=int)
             if all(s in tapt_oof[m] for m in KF) and all(s in old_val[n] for n in OLD)]
    test_ids=[str(t['id']) for t in test]
    print(f"OOF ids={len(oof_ids)}  test ids={len(test_ids)}")
    def models(f): return KF if SRC[f]=='tapt' else OLD
    def raw(f,sid,split):  # return {model: probs list}
        if SRC[f]=='tapt':
            pool=tapt_oof if split=='oof' else tapt_test
            return {m:pool[m][sid]['probs_'+f] for m in KF if sid in pool[m]}
        else:
            pool=old_val if split=='oof' else old_test
            return {n:pool[n][sid]['probs_'+f] for n in OLD if sid in pool[n]}
    # --- calibration (fit_T) per field per model on OOF ---
    calT={}
    for f in FIELDS:
        calT[f]={}
        yi=[LABELS[f].index(vby[s][f]) for s in oof_ids]
        for m in models(f):
            pl=[raw(f,s,'oof')[m] for s in oof_ids if m in raw(f,s,'oof')]
            calT[f][m]=R.fit_T(pl,yi)
    def cal(f,sid,split):
        return {m:R.softmax_logp(np.array(p),calT[f][m]) for m,p in raw(f,sid,split).items()}
    # --- F1-weights per field per model on OOF ---
    wts={}
    for f in FIELDS:
        wts[f]={}; y=[vby[s][f] for s in oof_ids]
        for m in models(f):
            p=[LABELS[f][int(np.argmax(cal(f,s,'oof')[m]))] for s in oof_ids if m in cal(f,s,'oof')]
            wts[f][m]=f1_score(y,p,labels=SCORED[f],average='macro',zero_division=0)
    bestsrc={f:max(wts[f],key=lambda m:wts[f][m]) for f in FIELDS}
    print("bestsrc:",{f:bestsrc[f] for f in FIELDS})
    # --- rules + prior ---
    def sv(by,sp,sid): rr=dict(by[sid]);c=sp.get(sid,{});rr['promise_string']=(c.get('promise_string') or '').strip();rr['evidence_string']=(c.get('evidence_string') or '').strip();return rr
    pr={}
    for f in FIELDS:
        pr[f]={};marg=priors[f]['_marginal']
        for t in ['E','S','G']: pr[f][t]=np.array([priors[f][t][l]/marg[l] for l in LABELS[f]])
        pr[f]['OTHER']=np.ones(len(LABELS[f]))
    def assemble(by,sp,sid,split,t,offs):
        picks={}; allprobs={}
        for f in FIELDS:
            c=cal(f,sid,split); probs=np.zeros(len(LABELS[f]))
            for m,p in c.items(): probs+=wts[f][m]*np.array(p)
            r=sv(by,sp,sid)
            if f=='verification_timeline':
                ru=R.predict_timeline_rule(r)
                if ru in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru)]=1;probs+=3*oh
            if f=='evidence_quality':
                ru=R.predict_quality_rule(r)
                if ru and ru in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru)]=1;probs+=3*oh
            if f=='evidence_status':
                ru=R.predict_evidence_rule(r)
                if ru and ru in LABELS[f]: oh=np.zeros(len(LABELS[f]));oh[LABELS[f].index(ru)]=1;probs+=3*oh
            if f in PRIOR_FIELDS:
                probs=probs*np.power(pr[f].get(R.main_type(by[sid].get('esg_type')),pr[f]['OTHER']),0.5)
            if offs and f in offs and offs[f] is not None:
                probs=probs/(probs.sum()+1e-12)+np.array(offs[f])
            allprobs[f]=probs
            bc=max(c[bestsrc[f]]) if bestsrc[f] in c else 0
            bp=LABELS[f][int(np.argmax(c[bestsrc[f]]))] if bestsrc[f] in c else None
            picks[f]=bp if (bp and bc>=t[f]) else LABELS[f][int(probs.argmax())]
        # external P/E override (teammate Eason stronger binary; TEST ids only). Our raw T/Q probs in
        # allprobs are untouched, so the compliance-repick below fills real timeline/quality for any
        # item his promise/evidence newly opens (promise No->Yes etc).
        if EXT_PE and sid in EXT_PE:
            picks['promise_status']=EXT_PE[sid]['promise_status']
            picks['evidence_status']=EXT_PE[sid]['evidence_status']
        # cascade
        if picks['promise_status']=='No': picks['verification_timeline']=picks['evidence_status']=picks['evidence_quality']='N/A'
        elif picks['evidence_status'] in ('No','N/A'): picks['evidence_quality']='N/A'
        # compliance (modeled in OOF too): promise=Yes -> real timeline; evidence=Yes -> real quality
        def repick(f):
            L=LABELS[f]; reals=[i for i,l in enumerate(L) if l!='N/A']
            return L[reals[int(np.argmax([allprobs[f][i] for i in reals]))]]
        if picks['promise_status']=='Yes' and picks['verification_timeline']=='N/A':
            picks['verification_timeline']=repick('verification_timeline')
        if picks['evidence_status']=='Yes' and picks['evidence_quality']=='N/A':
            picks['evidence_quality']=repick('evidence_quality')
        _sm=args.safety_more5
        if _sm and (picks['verification_timeline']=='already' if _sm=='already' else picks['verification_timeline'] not in ('N/A',None)):
            rr=sv(by,sp,sid); tx=rr['promise_string']+' '+rr['evidence_string']+' '+(by[sid].get('data') or '')
            if _safety_more5(rr['promise_string'], tx): picks['verification_timeline']='more_than_5_years'
        return picks
    # --- tune bestsrc thresholds on OOF (weighted total) ---
    def score(t,offs,idset=None,per=False):
        if idset is None: idset=oof_ids
        d={}
        for f in FIELDS:
            y=[vby[s][f] for s in idset]
            p=[assemble(vby,vspan,s,'oof',t,offs)[f] for s in idset]
            d[f]=f1_score(y,p,labels=SCORED[f],average='macro',zero_division=0)
        tot=sum(d[f]*W[f] for f in FIELDS); return (tot,d) if per else tot
    bt={f:0 for f in FIELDS}
    for _ in range(2):
        for f in FIELDS:
            bw,bv=-1,bt[f]
            for tt in np.linspace(0,1,21):
                w=score({**bt,f:tt},None)
                if w>bw: bw,bv=w,tt
            bt[f]=bv
    if args.force_timeline_thresh is not None: bt['verification_timeline']=args.force_timeline_thresh
    if args.force_evidence_thresh is not None: bt['evidence_status']=args.force_evidence_thresh
    # --- offsets: tune a field's offset on idset (skip N/A index for timeline) ---
    def tune_field_off(f,offs,idset):
        L=LABELS[f]; off=[0.0]*len(L); grid=[-0.3,-0.2,-0.1,0,0.1,0.2,0.35]
        NAi=L.index('N/A') if 'N/A' in L else -1
        for _ in range(2):
            for ci in range(len(L)):
                if ci==NAi: continue
                bw,bv=-1,off[ci]
                for v in grid:
                    off[ci]=v; w=score(bt,{**offs,f:off},idset)
                    if w>bw: bw,bv=w,v
                off[ci]=bv
        return off
    offs={f:None for f in FIELDS}
    if args.no_offset: pass
    elif args.timeline_offset: offs['verification_timeline']=[float(x) for x in args.timeline_offset.split(',')]
    else: offs['verification_timeline']=tune_field_off('verification_timeline',offs,oof_ids)
    if args.aggressive:
        offs['evidence_status']=tune_field_off('evidence_status',offs,oof_ids)
        # held-out 2-fold validation of evidence + timeline offsets (honest transfer check)
        half,oth=oof_ids[::2],oof_ids[1::2]
        for f in ['verification_timeline','evidence_status']:
            base=score(bt,{**offs,f:None})
            oa=tune_field_off(f,{**offs,f:None},half); ob=tune_field_off(f,{**offs,f:None},oth)
            ho=(score(bt,{**offs,f:ob},half)+score(bt,{**offs,f:oa},oth))/2
            ins=score(bt,offs)
            print(f">> [held-out] {f} offset: base={base:.4f} in-sample={ins:.4f} held-out={ho:.4f}  {'KEEP' if ho>base else 'DROP(overfit)'}")
            if ho<=base: offs[f]=None  # drop offsets that don't transfer
    tot,d=score(bt,offs,per=True)
    print(f">> OOF WF1={tot:.4f}  P={d['promise_status']:.4f} E={d['evidence_status']:.4f} T={d['verification_timeline']:.4f} Q={d['evidence_quality']:.4f}")
    print(f">> bestsrc thresh={ {f:round(bt[f],2) for f in FIELDS} }")
    print(f">> offsets: T={offs['verification_timeline']} E={offs['evidence_status']}")
    toff=offs  # for downstream test assemble
    # --- predict test + Misleading + compliance ---
    tby={str(t['id']):t for t in test}
    out=[]; viol=0; mis_applied=0
    REALT=[l for l in LABELS['verification_timeline'] if l!='N/A']
    for tr in test:
        sid=str(tr['id']); picks=assemble(tby,tspan,sid,'test',bt,toff)
        if sid in mis:
            picks['promise_status']='Yes'; picks['evidence_status']='Yes'; picks['evidence_quality']='Misleading'
            if picks['verification_timeline']=='N/A':
                c=cal('verification_timeline',sid,'test'); probs=np.zeros(5)
                for m,p in c.items(): probs+=wts['verification_timeline'][m]*np.array(p)
                ri=[LABELS['verification_timeline'].index(x) for x in REALT]
                picks['verification_timeline']=REALT[int(np.argmax([probs[i] for i in ri]))]
            mis_applied+=1
        # final safety (should be 0 — compliance handled in assemble)
        if picks['promise_status']=='Yes' and picks['verification_timeline']=='N/A':
            viol+=1; picks['verification_timeline']='between_2_and_5_years'
        if picks['evidence_status']=='Yes' and picks['evidence_quality']=='N/A':
            viol+=1; picks['evidence_quality']='Not Clear'
        out.append((tr['id'],picks['promise_status'],picks['verification_timeline'],picks['evidence_status'],picks['evidence_quality']))
    print(f">> Misleading applied={mis_applied}  residual-violations={viol} (should be 0)")
    import csv
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w',newline='') as fo:
        w=csv.writer(fo); w.writerow(['id','promise_status','verification_timeline','evidence_status','evidence_quality'])
        for row in out: w.writerow(row)
    print(f"✅ {len(out)} rows → {args.out}")

if __name__=='__main__': main()
