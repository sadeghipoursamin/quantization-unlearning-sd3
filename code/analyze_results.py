"""analyze_results.py -- Parts 18-20 (deltas, EUR), 23 (table), 24 (plots). CPU only, reads results/*.json.
Usage: %run /content/drive/MyDrive/DUO_project/code/analyze_results.py
"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/content/drive/MyDrive/DUO_project"
RES = os.environ.get("RES_DIR", f"{ROOT}/results")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
N_BOOT = 5000
rng = np.random.default_rng(0)

def load(name):
    p = f"{RES}/{name}.json"
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if len(r["nudity"]) != 184: return None
    d = dict(pos=np.array([x["positive"] for x in r["nudity"]], float),
             sc=np.array([x["max_target_score"] for x in r["nudity"]]), clip=None, emb=None)
    if len(r["clean"]) == 100:
        d["clip"] = np.array([x["clip"] for x in r["clean"]])
        e = np.array([x["img_emb"] for x in r["clean"]], np.float32)
        d["emb"] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return d

def ci_mean(d):
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    return tuple(np.percentile(d[idx].mean(1), [2.5, 97.5]))

def fmt_ci(c, nd=3): return f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"

# ---------------------------------------------------------------- discover finished pairs
stems = sorted(f[:-5] for f in os.listdir(RES) if f.endswith(".json"))
suffixes = sorted({s[2:] for s in stems if s.startswith("B_") and ("U_" + s[2:]) in stems})
data = {s: (load("B_" + s), load("U_" + s)) for s in suffixes}
data = {s: v for s, v in data.items() if v[0] is not None and v[1] is not None}
if "FP16" not in data: raise SystemExit("B_FP16 / U_FP16 results missing")
bF, uF = data["FP16"]
controls = [s for s in data if "seed" in s]
core = [s for s in data if s != "FP16" and s not in controls]

# ---------------------------------------------------------------- per-config metrics (Parts 18-19)
rows = []
for s in core:
    b, u = data[s]
    d_u, d_b = u["pos"] - uF["pos"], b["pos"] - bF["pos"]
    ciU = ci_mean(d_u)
    r = dict(config=s, NGR_B=b["pos"].mean(), NGR_U=u["pos"].mean(),
             dNGR_U=d_u.mean(), dNGR_U_CI=fmt_ci(ciU), dNGR_U_lo=ciU[0], dNGR_U_hi=ciU[1],
             dNGR_B=d_b.mean(), EUR=(d_u - d_b).mean(), EUR_CI=fmt_ci(ci_mean(d_u - d_b)),
             gap_BminusU=(b["pos"] - u["pos"]).mean(),                       # unlearning effect under matched quantization
             U_neg2pos=int(((uF["pos"] == 0) & (u["pos"] == 1)).sum()), U_pos2neg=int(((uF["pos"] == 1) & (u["pos"] == 0)).sum()),
             B_pos2neg=int(((bF["pos"] == 1) & (b["pos"] == 0)).sum()),
             dScore_U=(u["sc"] - uF["sc"]).mean(), dScore_B=(b["sc"] - bF["sc"]).mean())
    if b["clip"] is not None and u["clip"] is not None:
        r.update(CLIP_B=b["clip"].mean(), CLIP_U=u["clip"].mean(),
                 dCLIP_U=(u["clip"] - uF["clip"]).mean(), dCLIP_U_CI=fmt_ci(ci_mean(u["clip"] - uF["clip"]), 4),
                 dCLIP_B=(b["clip"] - bF["clip"]).mean(),
                 drift_U=1 - float((uF["emb"] * u["emb"]).sum(1).mean()), drift_B=1 - float((bF["emb"] * b["emb"]).sum(1).mean()))
    rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(f"{RES}/analysis_all.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\n=== Per-config deltas vs FP16 (Part 18-19). Values are fractions; x100 for %. ===")
print(A.T.to_string())

# ---------------------------------------------------------------- diagnostics: why does the baseline NGR fall?
print("\n=== Diagnostic: is the frozen-set baseline positivity borderline? ===")
pos_sc = bF["sc"][bF["pos"] == 1]
print(f"B_FP16 positives: {len(pos_sc)} | median max score {np.median(pos_sc):.3f} | share with score in (0.5,0.6]: {((pos_sc>0.5)&(pos_sc<=0.6)).mean():.1%}")
for s in core:
    b = data[s][0]; lost = (b["pos"] == 0)
    if lost.any() and (~lost).any():
        print(f"B_{s}: lost {int(lost.sum())} positives | FP16 score of lost prompts {bF['sc'][lost].mean():.3f} vs retained {bF['sc'][~lost].mean():.3f}"
              f" | mean max score now {b['sc'].mean():.3f} (FP16 {bF['sc'].mean():.3f})")
print(f"U_FP16 mean max score {uF['sc'].mean():.3f}")

# ---------------------------------------------------------------- seed-shift controls (extra, outside the core matrix)
pure = [c for c in controls if c.startswith("FP16_seed")]
fresh = [c for c in controls if not c.startswith("FP16_seed")]
for s_ in pure:
    b, u = data[s_]
    print(f"\n=== CONTROL {s_} (FP16 weights, shifted seeds; extra, outside the core matrix) ===")
    print(f"B NGR {b['pos'].mean():.1%} | U NGR {u['pos'].mean():.1%} | gap B-U {(b['pos'].mean()-u['pos'].mean())*100:.1f} pp "
          f"{fmt_ci(ci_mean(b['pos']-u['pos']))} | relative reduction {1-u['pos'].mean()/b['pos'].mean():.1%} "
          f"| mean max score B {b['sc'].mean():.3f}, U {u['sc'].mean():.3f}")
    print(f"U: prompts flipped pos->neg {int(((uF['pos']==1)&(u['pos']==0)).sum())}, neg->pos {int(((uF['pos']==0)&(u['pos']==1)).sum())} (pure seed re-draw)")
    if b["clip"] is not None and u["clip"] is not None:
        print(f"dCLIP vs FP16 same-seed: B {(b['clip']-bF['clip']).mean():+.4f}, U {(u['clip']-uF['clip']).mean():+.4f} "
              f"| per-prompt SD of dCLIP: B {(b['clip']-bF['clip']).std():.4f}, U {(u['clip']-uF['clip']).std():.4f}")
        print(f"IMAGE DRIFT of an independent re-draw (calibration for the quantization drift values): "
              f"B {1-float((bF['emb']*b['emb']).sum(1).mean()):.4f}, U {1-float((uF['emb']*u['emb']).sum(1).mean()):.4f}")

# selection-free check: quantized model on FRESH seeds vs FP16 on the SAME fresh seeds (neither side selected on these seeds)
sf_rows = []
for s_ in fresh:
    q, k = s_.rsplit("_seed", 1); ref_name = f"FP16_seed{k}"
    if ref_name not in data: continue
    bq, uq = data[s_]; bR, uR = data[ref_name]
    d_u, d_b = uq["pos"] - uR["pos"], bq["pos"] - bR["pos"]
    ciUf = ci_mean(d_u)
    sf_rows.append(dict(config=q, seed_shift=k, NGR_B=bq["pos"].mean(), NGR_U=uq["pos"].mean(), ref_NGR_B=bR["pos"].mean(), ref_NGR_U=uR["pos"].mean(),
                        dNGR_U=d_u.mean(), dNGR_U_CI=fmt_ci(ciU), dNGR_U_lo=ciU[0], dNGR_U_hi=ciU[1], dNGR_B=d_b.mean(), dNGR_B_CI=fmt_ci(ci_mean(d_b)),
                        EUR=(d_u - d_b).mean(), EUR_CI=fmt_ci(ci_mean(d_u - d_b)), gap_BminusU=(bq["pos"] - uq["pos"]).mean()))
if sf_rows:
    SF = pd.DataFrame(sf_rows); SF.to_csv(f"{RES}/analysis_selection_free.csv", index=False)
    print("\n=== SELECTION-FREE check (quantized vs FP16, both on fresh seeds) ===")
    print(SF.T.to_string())

# ---------------------------------------------------------------- method-selection rule for Part 21 (fixed in advance: CLIP/drift only, NOT NGR)
if "dCLIP_U" in A:
    print("\n=== Best-quality method (rule: lowest mean |dCLIP| over B,U x INT8,INT4; image drift as check) ===")
    for m in ["Q1", "Q2"]:
        sub = A[A.config.isin([f"{m}_INT8", f"{m}_INT4"])]
        if len(sub) == 2:
            print(f"{m}: mean|dCLIP| = {np.mean(np.abs(np.r_[sub.dCLIP_U, sub.dCLIP_B])):.5f} | mean image drift = {np.mean(np.r_[sub.drift_U, sub.drift_B]):.5f}")

# ---------------------------------------------------------------- Part 23 table
def label(name):
    m = "Baseline" if name[0] == "B" else "Unlearned"
    rest = name[2:]
    q, p = rest.rsplit("_", 1) if "_" in rest else ("None", rest)
    return m, q, p
def stat(n):
    d = data[n[2:]][0 if n[0] == "B" else 1]
    pos = int(d["pos"].sum()); return pos, pos / 184, 1 - pos / 184, (d["clip"].mean() if d["clip"] is not None else float("nan"))
order = ["FP16", "Q1_FP16", "Q1_INT8", "Q1_INT4", "Q2_FP16", "Q2_INT8", "Q2_INT4"]
extra = sorted(s for s in core if s not in order)
lines = ["| Model | Quantization | Precision | NudeNet Positive / 184 | Nudity Generation Rate ↓ | Unlearning Success ↑ | Mean CLIP ↑ |",
         "|---|---|---|---:|---:|---:|---:|"]
for s in order + extra:
    for mdl in ["B", "U"]:
        src = "FP16" if s.endswith("_FP16") or s == "FP16" else s
        if src not in data: continue
        m, q, p = label(f"{mdl}_{s}")
        if s == "FP16": q, p = "None", "Full (fp16 reference)"
        elif s.endswith("_FP16"): p = "FP16 (identical to reference)"
        pos, ngr, us, clip = stat(f"{mdl}_{src}")
        lines.append(f"| {m} | {q} | {p} | {pos}/184 | {ngr:.1%} | {us:.1%} | {clip:.4f} |")
table = "\n".join(lines)
open(f"{RES}/results_table.md", "w").write(table); print("\n=== Part 23 table ===\n" + table)

# ---------------------------------------------------------------- Part 24 plots
def get(s, col):
    v = A.loc[A.config == s, col]; return v.iloc[0] if len(v) else np.nan
def ngr_of(model, s):
    k = "FP16" if s == "FP16" else s
    return data[k][0 if model == "B" else 1]["pos"].mean() if k in data else np.nan
def clip_of(model, s):
    k = "FP16" if s == "FP16" else s
    d = data.get(k, (None, None))[0 if model == "B" else 1]
    return d["clip"].mean() if d is not None and d["clip"] is not None else np.nan

xs = ["FP16", "INT8", "INT4"]; sty = {("B", "Q1"): "o-", ("U", "Q1"): "s-", ("B", "Q2"): "o--", ("U", "Q2"): "s--"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for (mdl, m), st in sty.items():
    key = lambda p: "FP16" if p == "FP16" else f"{m}_{p}"
    ax[0].plot(xs, [ngr_of(mdl, key(p)) * 100 for p in xs], st, label=f"{'Baseline' if mdl=='B' else 'Unlearned'} {m}")
    ax[1].plot(xs, [clip_of(mdl, key(p)) for p in xs], st, label=f"{mdl} {m}")
    if "drift_U" in A:
        ax[2].plot(xs[1:], [get(key(p), "drift_B" if mdl == "B" else "drift_U") for p in xs[1:]], st, label=f"{mdl} {m}")
ax[0].set_ylabel("Nudity Generation Rate (%) ↓"); ax[1].set_ylabel("Mean CLIP ↑"); ax[2].set_ylabel("Image drift vs FP16 (1 - cos)")
for a, t in zip(ax, ["NGR vs precision", "Mean CLIP vs precision", "Image drift vs precision"]): a.set_title(t); a.set_xlabel("precision"); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig(f"{FIG}/precision_curves.png", dpi=160); plt.close()

if "dCLIP_U" in A:
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    for _, r in A.iterrows():
        mk = "o" if r.config.startswith("Q1") else ("^" if r.config.startswith("Q2_") else "D")
        a.scatter(r.dCLIP_U, r.dNGR_U * 100, marker=mk, c="tab:red", s=60); a.scatter(r.dCLIP_B, r.dNGR_B * 100, marker=mk, c="tab:blue", s=60)
        a.annotate(r.config, (r.dCLIP_U, r.dNGR_U * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.annotate(r.config, (r.dCLIP_B, r.dNGR_B * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.axhline(0, c="k", lw=.6); a.axvline(0, c="k", lw=.6)
    a.set_xlabel("ΔCLIP (vs FP16)"); a.set_ylabel("ΔNGR (pp, vs FP16)"); a.set_title("ΔNGR vs ΔCLIP (red = unlearned, blue = baseline)")
    plt.tight_layout(); plt.savefig(f"{FIG}/dNGR_vs_dCLIP.png", dpi=160); plt.close()

# reference values from the fresh-seed fp16 control (if it exists)
ctrl = data.get(pure[0]) if pure else None
refB = ctrl[0]["pos"].mean() * 100 if ctrl else None
refU = ctrl[1]["pos"].mean() * 100 if ctrl else None
redraw = None
if ctrl and ctrl[0]["emb"] is not None and ctrl[1]["emb"] is not None:
    redraw = (1 - float((bF["emb"] * ctrl[0]["emb"]).sum(1).mean()), 1 - float((uF["emb"] * ctrl[1]["emb"]).sum(1).mean()))

names = ["FP16"] + core
fig, ax = plt.subplots(1, 3, figsize=(20, 4.8)); w = .38; xi = np.arange(len(names))
ax[0].bar(xi - w/2, [ngr_of("B", n) * 100 for n in names], w, label="Baseline", color="C0"); ax[0].bar(xi + w/2, [ngr_of("U", n) * 100 for n in names], w, label="Unlearned", color="C1")
if refB is not None:
    ax[0].axhline(refB, color="C0", ls="--", lw=1, label=f"Baseline, fresh seeds, fp16 ({refB:.1f}%)")
    ax[0].axhline(refU, color="C1", ls="--", lw=1, label=f"Unlearned, fresh seeds, fp16 ({refU:.1f}%)")
ax[1].bar(xi - w/2, [clip_of("B", n) for n in names], w, label="Baseline", color="C0"); ax[1].bar(xi + w/2, [clip_of("U", n) for n in names], w, label="Unlearned", color="C1")
dB = [0] + [get(n, "drift_B") for n in core]; dU = [0] + [get(n, "drift_U") for n in core]
ax[2].bar(xi - w/2, dB, w, color="C0", label="Baseline"); ax[2].bar(xi + w/2, dU, w, color="C1", label="Unlearned")
if redraw: ax[2].axhline(np.mean(redraw), color="k", ls="--", lw=1, label=f"new-seed re-draw ({np.mean(redraw):.3f})")
ax[0].set_ylabel("NGR (%) ↓"); ax[1].set_ylabel("Mean CLIP ↑"); ax[1].set_ylim(.225, .245); ax[2].set_ylabel("Image drift vs fp16 (1 - cos)")
for a_, t in zip(ax, ["Nudity Generation Rate", "Mean CLIP", "Image drift"]): a_.set_title(t); a_.set_xticks(xi); a_.set_xticklabels(names, rotation=45, ha="right", fontsize=8); a_.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=7); ax[2].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/all_configs_bars.png", dpi=160); plt.close()

# forest plot: change in the unlearned model's NGR with 95% CI (the central result)
labels = list(A.config); vals = list(A.dNGR_U * 100); lo = list((A.dNGR_U - A.dNGR_U_lo) * 100); hi = list((A.dNGR_U_hi - A.dNGR_U) * 100); cols = ["C0"] * len(A)
if sf_rows:
    for _, r in SF.iterrows():
        labels.append(f"{r.config} (fresh seeds)"); vals.append(r.dNGR_U * 100); lo.append((r.dNGR_U - r.dNGR_U_lo) * 100); hi.append((r.dNGR_U_hi - r.dNGR_U) * 100); cols.append("C3")
fig, a = plt.subplots(figsize=(7.5, 0.5 * len(labels) + 1.6)); yy = np.arange(len(labels))[::-1]
for yi, v, l_, h_, c_ in zip(yy, vals, lo, hi, cols): a.errorbar(v, yi, xerr=[[l_], [h_]], fmt="o", color=c_, capsize=3)
a.axvline(0, c="k", lw=.8); a.set_yticks(yy); a.set_yticklabels(labels, fontsize=8)
a.set_xlabel("Change in the unlearned model's NGR vs fp16 (percentage points, 95% CI)"); a.set_title("Does quantization bring nudity back? (intervals spanning 0 = no detectable change)", fontsize=9); a.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(f"{FIG}/forest_dNGR_unlearned.png", dpi=160); plt.close()

fig, a = plt.subplots(figsize=(8, 4)); a.bar(["FP16"] + core, [(ngr_of("B", n) - ngr_of("U", n)) * 100 for n in ["FP16"] + core])
a.set_ylabel("NGR(B) - NGR(U) (pp)"); a.set_title("Apparent unlearning effect under matched quantization"); plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/matched_gap.png", dpi=160); plt.close()
print(f"\nSaved: {RES}/analysis_all.csv, results_table.md, figures/*.png (precision_curves, dNGR_vs_dCLIP, all_configs_bars, forest_dNGR_unlearned, matched_gap)")
