"""Analytical cascade model for VeriPromiseESG (N/A-excluded hierarchical macro-F1).

Scoring (verified in repo: sklearn f1_score(labels=SCORED[f], average='macro')):
  promise  2-class macro {Yes,No}                        w=.20
  evidence 3-class macro {Yes,No,N/A}  (N/A IS scored)   w=.30
  quality  3-class macro {Clear,NotClear,Misleading}     w=.35  (N/A excluded, labels-restricted)
  timeline 4-class macro {within2y,b2_5y,more5,already}  w=.15  (N/A excluded)

Labels-restricted macro semantics: pred real-class on GT-N/A row => FP for that class
(precision damage); pred N/A on GT-real row => FN only (recall damage), never an FP.

Population (train-1000 proportions, assumed for test):
  P: Yes .814 / No .186
  E: Yes .677 / No .137 / NA .186
  T: already .366 / b2_5y .238 / more5 .197 / within2y .013 / NA .186
  Q: Clear .552 / NC .124 / Mis .001 / NA .323
"""

W = dict(P=.20, E=.30, Q=.35, T=.15)
pYes, pNo = .814, .186
mEY, mEN, mENA = .677, .137, .186
TCLS = ['already','b2_5y','more5','within2y']
mT = dict(already=.366, b2_5y=.238, more5=.197, within2y=.013)
QCLS = ['Clear','NC']
mQ = dict(Clear=.552, NC=.124)          # Mis (.001) handled as constant F1
ATTR = dict(already=.55, b2_5y=.20, more5=.20, within2y=.05)  # confusion attractor / u
V = dict(Clear=.85, NC=.15)             # quality module preds on falsely-open rows
QY_OPEN = .70                           # P(evidence-module says Yes | falsely-open P=No row)

def f1(tp, fn, fp):
    return 2*tp/(2*tp+fn+fp) if (2*tp+fn+fp) > 0 else 0.0

def promise_field(rho, phi):
    f1Y = f1(pYes*rho, pYes*(1-rho), pNo*phi)
    f1N = f1(pNo*(1-phi), pNo*phi, pYes*(1-rho))
    return (f1Y+f1N)/2, (f1Y, f1N)

def evidence_field(rho, phi, eYY, eNN, qy=QY_OPEN):
    # predictions: GT EY: NA(1-rho) | Y rho*eYY | N rho*(1-eYY)
    #              GT EN: NA(1-rho) | N rho*eNN | Y rho*(1-eNN)
    #              GT ENA: NA(1-phi)| Y phi*qy  | N phi*(1-qy)
    tpY = mEY*rho*eYY; fnY = mEY*(1-rho*eYY); fpY = mEN*rho*(1-eNN) + mENA*phi*qy
    tpN = mEN*rho*eNN; fnN = mEN*(1-rho*eNN); fpN = mEY*rho*(1-eYY) + mENA*phi*(1-qy)
    tpA = mENA*(1-phi); fnA = mENA*phi;       fpA = (mEY+mEN)*(1-rho)
    return (f1(tpY,fnY,fpY)+f1(tpN,fnN,fpN)+f1(tpA,fnA,fpA))/3

def t_conf(tvec):
    """4x4 row-stochastic conditional confusion: diag t_c, off-mass ~ ATTR."""
    Tm = {}
    for c in TCLS:
        row = {}
        off = 1-tvec[c]
        denom = sum(ATTR[c2] for c2 in TCLS if c2 != c)
        for c2 in TCLS:
            row[c2] = tvec[c] if c2 == c else off*ATTR[c2]/denom
        Tm[c] = row
    return Tm

def timeline_field(rho, phi, Tm, per_class=False):
    out = {}
    for c in TCLS:
        tp = mT[c]*rho*Tm[c][c]
        fn = mT[c]*(1-rho*Tm[c][c])
        fp = sum(mT[c2]*rho*Tm[c2][c] for c2 in TCLS if c2 != c) + .186*phi*ATTR[c]/sum(ATTR.values())
        out[c] = f1(tp, fn, fp)
    mac = sum(out.values())/4
    return (mac, out) if per_class else mac

def quality_field(rho, phi, eYY, eNN, rvec, f1_mis=1/3, qy=QY_OPEN, per_class=False):
    gate_real = rho*eYY                       # open prob on GT Clear/NC rows
    fp_openEN = mEN*rho*(1-eNN)               # GT (P=Y,E=N) falsely open for quality
    fp_openNA = mENA*phi*qy                   # GT P=No falsely open (evid says Yes)
    out = {}
    for c in QCLS:
        oth = [x for x in QCLS if x != c][0]
        tp = mQ[c]*gate_real*rvec[c]
        fn = mQ[c]*(1-gate_real*rvec[c])
        fp = mQ[oth]*gate_real*(1-rvec[oth]) + (fp_openEN+fp_openNA)*V[c]
        out[c] = f1(tp, fn, fp)
    mac = (out['Clear']+out['NC']+f1_mis)/3
    return (mac, out) if per_class else mac

def total(rho, phi, eYY, eNN, tvec, rvec, f1_mis=1/3):
    P,_ = promise_field(rho, phi)
    E = evidence_field(rho, phi, eYY, eNN)
    T = timeline_field(rho, phi, t_conf(tvec))
    Q = quality_field(rho, phi, eYY, eNN, rvec, f1_mis)
    S = W['P']*P + W['E']*E + W['T']*T + W['Q']*Q
    return S, dict(P=P, E=E, T=T, Q=Q)

# ---------- fitting helpers (1-D bisection on a scalar) ----------
def fit_scalar(fun, target, lo, hi, iters=80):
    """Find s in [min(lo,hi),max(lo,hi)] with fun(s)=target; fun monotonic (auto-orient)."""
    a, b = min(lo, hi), max(lo, hi)
    fa, fb = fun(a), fun(b)
    if fa > fb:   # make fun increasing on [a,b]
        a, b, fa, fb = b, a, fb, fa
    for _ in range(iters):
        mid = (a+b)/2
        if fun(mid) < target: a = mid
        else: b = mid
    return (a+b)/2

# --- promise error-direction anchor: A-era val No-recall ~.59 => phi_A=.41, macro .7582
phiA = .41
rhoA = fit_scalar(lambda r: -promise_field(r, phiA)[0], -.7582, .999, .80)
_e = [1-rhoA, phiA]; _s = _e[0]+_e[1]
errP = [_e[0]/_s, _e[1]/_s]   # error-direction unit vector
def promise_params(macro_target):
    s = fit_scalar(lambda s: -promise_field(1-s*errP[0], s*errP[1])[0], -macro_target, 1.5, .0)
    return 1-s*errP[0], s*errP[1]

# --- evidence anchor: base guesses eYY0=.90, eNN0=.45; scale error jointly
def evidence_params(macro_target, rho, phi):
    e0 = [1-.90, 1-.45]
    s = fit_scalar(lambda s: -evidence_field(rho, phi, 1-s*e0[0], 1-s*e0[1]), -macro_target, 2.5, .0)
    return 1-s*e0[0], 1-s*e0[1]

# ============ FIT baseline = 0.6540 config (P .7759 E .6708 T .5907 Q .5971) ============
rho0, phi0 = promise_params(.7759)
eYY0, eNN0 = evidence_params(.6708, rho0, phi0)
tshape = dict(already=.80, b2_5y=.62, more5=.58, within2y=.30)
st = fit_scalar(lambda s: -timeline_field(rho0, phi0, t_conf({c: min(.995, s*tshape[c]) for c in TCLS})),
                -.5907, 2.2, .2)
tvec0 = {c: min(.995, st*tshape[c]) for c in TCLS}
rshape = dict(Clear=.88, NC=.45)
sq = fit_scalar(lambda s: -quality_field(rho0, phi0, eYY0, eNN0,
                {c: min(.995, s*rshape[c]) for c in QCLS}), -.5971, 2.2, .2)
rvec0 = {c: min(.995, sq*rshape[c]) for c in QCLS}

S0, F0 = total(rho0, phi0, eYY0, eNN0, tvec0, rvec0)
print("=== BASELINE FIT (target 0.6540: .7759/.6708/.5907/.5971) ===")
print(f"rho={rho0:.4f} phi={phi0:.4f} eYY={eYY0:.4f} eNN={eNN0:.4f}")
print(f"tvec={ {c: round(v,3) for c,v in tvec0.items()} }  rvec={ {c: round(v,3) for c,v in rvec0.items()} }")
print(f"S={S0:.4f}  fields={ {k: round(v,4) for k,v in F0.items()} }")

# ============ PREDICT: strong4 (fit P .8199, E .6878 -> predict T,Q) ============
def predict(macroP, macroE, label, obsT=None, obsQ=None, f1_mis=1/3):
    r, p = promise_params(macroP)
    eY, eN = evidence_params(macroE, r, p)
    T = timeline_field(r, p, t_conf(tvec0))
    Q = quality_field(r, p, eY, eN, rvec0, f1_mis)
    S, F = total(r, p, eY, eN, tvec0, rvec0, f1_mis)
    print(f"\n--- {label}: fit P={macroP} E={macroE} (rho={r:.4f} phi={p:.4f} eYY={eY:.4f} eNN={eN:.4f})")
    print(f"pred T={T:.4f} (obs {obsT})   pred Q={Q:.4f} (obs {obsQ})   pred S={S:.4f}")
    # cascade-only share of E change (rho/phi move, e's held at baseline):
    E_gate_only = evidence_field(r, p, eYY0, eNN0)
    print(f"   E if only gate changed (e's fixed): {E_gate_only:.4f}  (baseline E {F0['E']:.4f})")
    return r, p, eY, eN, T, Q, S

print("\n=== PREDICTIVE VALIDATION (downstream confusions FROZEN at baseline fit) ===")
rs4, ps4, eYs4, eNs4, *_ = predict(.8199, .6878, "strong4", obsT=.6091, obsQ=.6122)
predict(.7819, .6804, "seed777", obsT=.5931, obsQ=.6095)
predict(.7582, .6354, "A-config (expect +residual: older/weaker downstream)", obsT=.5836, obsQ=.4303, f1_mis=0.0)
predict(.7884, .6773, "teammate Eason (no Mis)", obsT=.588, obsQ=.4412, f1_mis=0.0)

# ============ ORACLE BOUNDS (val-predictable now) ============
print("\n=== ORACLE PREDICTIONS (from baseline-fitted downstream) ===")
for lab, (r, p, eY, eN) in dict(
    O_P=(1.0, 0.0, eYY0, eNN0),
    O_E=(rho0, phi0, 1.0, 1.0),
    O_PE=(1.0, 0.0, 1.0, 1.0),
).items():
    S, F = total(r, p, eY, eN, tvec0, rvec0)
    if lab == 'O_E':  # GT evidence: E field = 1; quality FP sources need care -> eYY=eNN=1 handles it, but E macro:
        E = 1.0
        S = W['P']*F['P'] + W['E']*1.0 + W['T']*F['T'] + W['Q']*F['Q']
        F = dict(F, E=1.0)
    if lab == 'O_PE':
        S = W['P']*1.0 + W['E']*1.0 + W['T']*F['T'] + W['Q']*F['Q']
        F = dict(F, P=1.0, E=1.0)
    if lab == 'O_P':
        Pf = 1.0
        S = W['P']*1.0 + W['E']*F['E'] + W['T']*F['T'] + W['Q']*F['Q']
        F = dict(F, P=1.0)
    print(f"{lab}: S={S:.4f}  fields={ {k: round(v,4) for k,v in F.items()} }")

# ============ EFFECTIVE WEIGHT of promise & evidence at strong4 point ============
print("\n=== EFFECTIVE WEIGHTS (numerical dS at strong4 operating point) ===")
def S_of_rho(r):
    S, _ = total(r, ps4, eYs4, eNs4, tvec0, rvec0); return S
def MP_of_rho(r):
    return promise_field(r, ps4)[0]
d = 1e-4
dS = (S_of_rho(rs4+d)-S_of_rho(rs4-d))/(2*d)
dMP = (MP_of_rho(rs4+d)-MP_of_rho(rs4-d))/(2*d)
print(f"dS/drho={dS:.4f}  dM_P/drho={dMP:.4f}  => W_eff(promise via recall) = {dS/dMP:.4f}  (nominal 0.20)")
def S_of_phi(p):
    S, _ = total(rs4, p, eYs4, eNs4, tvec0, rvec0); return S
def MP_of_phi(p):
    return promise_field(rs4, p)[0]
dSp = (S_of_phi(ps4-d)-S_of_phi(ps4+d))/(2*d)   # improving = decreasing phi
dMPp = (MP_of_phi(ps4-d)-MP_of_phi(ps4+d))/(2*d)
print(f"dS/d(-phi)={dSp:.4f} dM_P/d(-phi)={dMPp:.4f} => W_eff(promise via specificity) = {dSp/dMPp:.4f}")
# evidence effective weight (vary eNN & eYY jointly along fitted direction)
def S_of_eYY(e):
    S, _ = total(rs4, ps4, e, eNs4, tvec0, rvec0); return S
def ME_of_eYY(e):
    return evidence_field(rs4, ps4, e, eNs4)
dSe = (S_of_eYY(eYs4+d)-S_of_eYY(eYs4-d))/(2*d)
dMEe = (ME_of_eYY(eYs4+d)-ME_of_eYY(eYs4-d))/(2*d)
print(f"W_eff(evidence via eYY) = {dSe/dMEe:.4f}  (nominal 0.30)")
def S_of_eNN(e):
    S, _ = total(rs4, ps4, eYs4, e, tvec0, rvec0); return S
def ME_of_eNN(e):
    return evidence_field(rs4, ps4, eYs4, e)
dSn = (S_of_eNN(eNs4+d)-S_of_eNN(eNs4-d))/(2*d)
dMEn = (ME_of_eNN(eNs4+d)-ME_of_eNN(eNs4-d))/(2*d)
print(f"W_eff(evidence via eNN) = {dSn/dMEn:.4f}  (nominal 0.30)")

# ============ PER-ITEM COST of one promise FN, by downstream GT type ============
print("\n=== COST OF ONE PROMISE FN (total-score points x1e4, test N=2000) ===")
N = 2000.0
def S_counts(perturb=None):
    """Rebuild S from expected counts; perturb: dict of (field,class)->(dtp,dfn,dfp) in items."""
    r, p, eY, eN = rs4, ps4, eYs4, eNs4
    # promise counts
    C = {}
    C[('P','Yes')] = [N*pYes*r, N*pYes*(1-r), N*pNo*p]
    C[('P','No')]  = [N*pNo*(1-p), N*pNo*p, N*pYes*(1-r)]
    C[('E','Yes')] = [N*mEY*r*eY, N*mEY*(1-r*eY), N*(mEN*r*(1-eN)+mENA*p*QY_OPEN)]
    C[('E','No')]  = [N*mEN*r*eN, N*mEN*(1-r*eN), N*(mEY*r*(1-eY)+mENA*p*(1-QY_OPEN))]
    C[('E','NA')]  = [N*mENA*(1-p), N*mENA*p, N*(mEY+mEN)*(1-r)]
    Tm = t_conf(tvec0)
    for c in TCLS:
        tp = N*mT[c]*r*Tm[c][c]; fn = N*mT[c]*(1-r*Tm[c][c])
        fp = N*(sum(mT[c2]*r*Tm[c2][c] for c2 in TCLS if c2 != c) + .186*p*ATTR[c]/sum(ATTR.values()))
        C[('T',c)] = [tp, fn, fp]
    gate = r*eY; fpo = N*(mEN*r*(1-eN)+mENA*p*QY_OPEN)
    for c in QCLS:
        oth = [x for x in QCLS if x != c][0]
        C[('Q',c)] = [N*mQ[c]*gate*rvec0[c], N*mQ[c]*(1-gate*rvec0[c]),
                      N*mQ[oth]*gate*(1-rvec0[oth]) + fpo*V[c]]
    if perturb:
        for k, (dtp, dfn, dfp) in perturb.items():
            C[k][0] += dtp; C[k][1] += dfn; C[k][2] += dfp
    Pf = (f1(*C[('P','Yes')])+f1(*C[('P','No')]))/2
    Ef = (f1(*C[('E','Yes')])+f1(*C[('E','No')])+f1(*C[('E','NA')]))/3
    Tf = sum(f1(*C[('T',c)]) for c in TCLS)/4
    Qf = (f1(*C[('Q','Clear')])+f1(*C[('Q','NC')])+1/3)/3
    return W['P']*Pf+W['E']*Ef+W['T']*Tf+W['Q']*Qf

base = S_counts()
r, eY = rs4, eYs4
for tc in TCLS:
    for qc, has_ev in [('Clear', True)]:
        pert = {}
        # promise: one open GT-Yes item (was expected-correct) -> forced No
        pert[('P','Yes')] = (-1, +1, 0)
        pert[('P','No')]  = (0, 0, +1)
        # evidence GT Yes (has_ev): open contribution (eY TP, (1-eY) FP_No) -> closed: FN_Yes+1, FP_NA+1
        pert[('E','Yes')] = (-eY, +eY, 0)
        pert[('E','No')]  = (0, 0, -(1-eY))
        pert[('E','NA')]  = (0, 0, +1)
        # timeline GT tc: open expected Tm[tc][:] -> closed FN
        Tm = t_conf(tvec0)
        dtp = -Tm[tc][tc]
        pert[('T',tc)] = (dtp, -dtp, 0)
        for c2 in TCLS:
            if c2 != tc:
                old = pert.get(('T',c2), (0,0,0))
                pert[('T',c2)] = (old[0], old[1], old[2]-Tm[tc][c2])
        # quality GT Clear, gate was open w.p. eY given promise open
        pert[('Q','Clear')] = (-eY*rvec0['Clear'], +eY*rvec0['Clear'], 0)
        pert[('Q','NC')] = (0, 0, -eY*(1-rvec0['Clear']))
        cost = (base - S_counts(pert))*1e4
        print(f"  GT timeline={tc:9s} (E=Yes,Q=Clear): {cost:7.2f} x1e-4 total points")
# nominal-only accounting for comparison
pert = {('P','Yes'): (-1, +1, 0), ('P','No'): (0, 0, +1)}
print(f"  promise-field-only (nominal) cost:      {(base - S_counts(pert))*1e4:7.2f} x1e-4")

# per-FP (false open) cost: one GT-No item flips No->Yes
pert = {('P','No'): (-1, +1, 0), ('P','Yes'): (0, 0, +1),
        ('E','NA'): (-1, +1, 0),
        ('E','Yes'): (0, 0, +QY_OPEN), ('E','No'): (0, 0, +(1-QY_OPEN))}
au = {c: ATTR[c]/sum(ATTR.values()) for c in TCLS}
for c in TCLS:
    old = pert.get(('T',c), (0,0,0)); pert[('T',c)] = (old[0], old[1], old[2]+au[c])
pert[('Q','Clear')] = (0, 0, +QY_OPEN*V['Clear']); pert[('Q','NC')] = (0, 0, +QY_OPEN*V['NC'])
print(f"  cost of one promise FP (false open):    {(base - S_counts(pert))*1e4:7.2f} x1e-4")

# ============ COMPONENT BREAKDOWN of dS/drho (where the effective weight lives) ============
print("\n=== dS/drho COMPONENTS at strong4 point ===")
d = 1e-4
for name, w, fn in [
    ('P', W['P'], lambda r: promise_field(r, ps4)[0]),
    ('E', W['E'], lambda r: evidence_field(r, ps4, eYs4, eNs4)),
    ('T', W['T'], lambda r: timeline_field(r, ps4, t_conf(tvec0))),
    ('Q', W['Q'], lambda r: quality_field(r, ps4, eYs4, eNs4, rvec0)),
]:
    g = (fn(rs4+d)-fn(rs4-d))/(2*d)
    print(f"  {name}: dM/drho={g:+.4f}  weighted={w*g:+.4f}")

# ============ VALUE OF FIXING 10 ERRORS OF EACH TYPE (prescription table) ============
print("\n=== POINTS GAINED PER 10 ERRORS FIXED (x1e-4 total points, test N=2000) ===")
Tm = t_conf(tvec0)
au = {c: ATTR[c]/sum(ATTR.values()) for c in TCLS}
def val10(pert, k=10):
    p = {kk: tuple(k*x for x in v) for kk, v in pert.items()}
    return (S_counts(p) - base)*1e4

fixes = {}
# 1/2: promise FN repair (modal item: T=already,E=Yes,Q=Clear) / promise FP repair
fnfix = {('P','Yes'): (+1,-1,0), ('P','No'): (0,0,-1),
         ('E','Yes'): (+eYs4,-eYs4,0), ('E','No'): (0,0,+(1-eYs4)), ('E','NA'): (0,0,-1),
         ('T','already'): (+Tm['already']['already'],-Tm['already']['already'],0),
         ('Q','Clear'): (+eYs4*rvec0['Clear'],-eYs4*rvec0['Clear'],0),
         ('Q','NC'): (0,0,+eYs4*(1-rvec0['Clear']))}
for c2 in TCLS:
    if c2!='already':
        old=fnfix.get(('T',c2),(0,0,0)); fnfix[('T',c2)]=(old[0],old[1],old[2]+Tm['already'][c2])
fixes['promise FN -> Yes (modal item)'] = val10(fnfix)
fnfix_w2 = dict(fnfix)
fnfix_w2[('T','within2y')] = (+Tm['within2y']['within2y'],-Tm['within2y']['within2y'],0)
fnfix_w2[('T','already')] = (0,0,+Tm['within2y']['already'])
for c2 in ['b2_5y','more5']:
    fnfix_w2[('T',c2)] = (0,0,+Tm['within2y'][c2])
fixes['promise FN -> Yes (within2y item)'] = val10(fnfix_w2)
fpfix = {('P','No'): (+1,-1,0), ('P','Yes'): (0,0,-1), ('E','NA'): (+1,-1,0),
         ('E','Yes'): (0,0,-QY_OPEN), ('E','No'): (0,0,-(1-QY_OPEN))}
for c in TCLS:
    old=fpfix.get(('T',c),(0,0,0)); fpfix[('T',c)]=(old[0],old[1],old[2]-au[c])
fpfix[('Q','Clear')]=(0,0,-QY_OPEN*V['Clear']); fpfix[('Q','NC')]=(0,0,-QY_OPEN*V['NC'])
fixes['promise FP -> No (false open)'] = val10(fpfix)
# 3/4: evidence within-gate fixes
e34 = {('E','Yes'): (+1,-1,0), ('E','No'): (0,0,-1),
       ('Q','Clear'): (+rvec0['Clear'],-rvec0['Clear'],0), ('Q','NC'): (0,0,+(1-rvec0['Clear']))}
fixes['evidence Yes-miss fixed (GT Clear row)'] = val10(e34)
e43 = {('E','No'): (+1,-1,0), ('E','Yes'): (0,0,-1),
       ('Q','Clear'): (0,0,-V['Clear']), ('Q','NC'): (0,0,-V['NC'])}
fixes['evidence No-miss fixed'] = val10(e43)
# 5/6: timeline conditional fixes
for c in ['already','within2y']:
    tf = {('T',c): (+1,-1,0)}
    for c2 in TCLS:
        if c2!=c:
            tf[('T',c2)]=(0,0,-Tm[c][c2]/(1-Tm[c][c]) * (1-Tm[c][c]) if False else (0,0,-ATTR[c2]/sum(ATTR[x] for x in TCLS if x!=c)))
    tf = {('T',c): (+1,-1,0)}
    den = sum(ATTR[x] for x in TCLS if x!=c)
    for c2 in TCLS:
        if c2!=c: tf[('T',c2)]=(0,0,-ATTR[c2]/den)
    fixes[f'timeline conditional fix ({c})'] = val10(tf)
# 7/8: quality conditional fixes
fixes['quality Clear-miss fixed'] = val10({('Q','Clear'): (+1,-1,0), ('Q','NC'): (0,0,-1)})
fixes['quality NC-miss fixed'] = val10({('Q','NC'): (+1,-1,0), ('Q','Clear'): (0,0,-1)})
for k,v in fixes.items():
    print(f"  {k:42s}: {v:8.2f}")
# 9: one true Misleading caught (support 2 scenario: F1 1/3 -> 1/2)
print(f"  {'ONE true Misleading caught (F1 .33->.50)':42s}: {0.35*(0.5-1/3)/3*1e4:8.2f}  (single error!)")

# ============ DELTAS TABLE (predicted vs observed, vs 0.6540 baseline) ============
print("\n=== PRED vs OBS DELTAS (fields, relative to 0.6540 baseline) ===")
rows = [("strong4", .8199,.6878,1/3, .0184,.0151,.0220),
        ("seed777", .7819,.6804,1/3, .0024,.0124,.0088)]
for lab,mp,me,fm,dTo,dQo,dSo in rows:
    r,p = promise_params(mp); eY,eN = evidence_params(me,r,p)
    T = timeline_field(r,p,t_conf(tvec0)); Q = quality_field(r,p,eY,eN,rvec0,fm)
    S,_ = total(r,p,eY,eN,tvec0,rvec0,fm)
    print(f"  {lab}: dT pred {T-F0['T']:+.4f} obs {dTo:+.4f} | dQ pred {Q-F0['Q']:+.4f} obs {dQo:+.4f} | dS pred {S-S0:+.4f} obs {dSo:+.4f}")
