"""
LVM-AI Regressie Analyse – Continue LVM schatting bij atleten
- Spearman correlatie, gedeelde assen per eenheid, LVH-kleurcodering
- Bland-Altman in absolute (g) en relatieve (%) eenheden
- OLS lineaire kalibratie als primair, Passing-Bablok als sensitivity
- Sign-conventie: CMR - Predicted (paper-conventie: neg = overschatting)
- 95% CI rond rho en MAE in scatter-titels (bootstrap, 1000 iter)
- Legenda overal rechts-onder, alle figuren openen in een keer
Gebruik: python validate_regression.py
"""

import subprocess, sys

def install(packages):
    for pkg in packages:
        try:
            __import__(pkg.split("==")[0].replace("-","_").replace("beautifulsoup4","bs4"))
        except ImportError:
            print(f"Installeren: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

install([
    "tensorflow==2.19.0",
    "beautifulsoup4", "lxml",
    "numpy", "pandas", "scikit-learn",
    "matplotlib", "seaborn", "scipy",
])

import os
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---------------------------------------------------------
MODEL_FILE = "ecg_rest_raw_age_sex_bmi_lvm_asymmetric_loss.h5"
ECG_DIR    = "ecg_bestanden"
LABELS_CSV = "regressionlabels.csv"
OUTPUT_DIR = "resultaten"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ECG_NORM = 2000.0
AGE_MEAN, AGE_STD = 63.35798891483556, 7.554638350423902
BMI_MEAN, BMI_STD = 27.3397, 4.7721
LVM_MEAN, LVM_STD = 89.70372484725051, 24.803669503436304
LEAD_ORDER  = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
ECG_SAMPLES = 5000
LVH_COLOR_MAP = [(1, "crimson", "LVH+"), (0, "steelblue", "LVH-")]
MARGIN = 0.05
LEGEND_LOC = "lower right"

def boxplot_compat(ax, data, labels, **kwargs):
    try:    return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError: return ax.boxplot(data, labels=labels, **kwargs)

def data_lims(*arrays, margin=MARGIN):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0: return 0, 1
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * margin if hi > lo else 1.0
    return lo - pad, hi + pad

def pct_diffs(pred, truth):
    """Relatieve verschillen in %: 100*(truth - pred)/mean(pred, truth) - paper-conventie."""
    pred = np.asarray(pred, dtype=float); truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[mask], truth[mask]
    mean_v = (pred + truth) / 2
    safe = np.abs(mean_v) > 1e-9
    return 100.0 * (truth[safe] - pred[safe]) / mean_v[safe]

def passing_bablok(x, y):
    """Passing-Bablok regression: non-parametrische method-comparison.
       Retourneert (slope, intercept). O(n^2)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                s = (y[j] - y[i]) / (x[j] - x[i])
                if s != -1:
                    slopes.append(s)
    slopes = np.sort(np.array(slopes, dtype=float))
    N = len(slopes)
    if N == 0:
        return np.nan, np.nan
    K = int(np.sum(slopes < -1))
    if N % 2 == 1:
        idx = (N - 1) // 2 + K
        idx = max(0, min(idx, N - 1))
        slope = float(slopes[idx])
    else:
        i1 = max(0, min(N // 2 - 1 + K, N - 1))
        i2 = max(0, min(N // 2 + K,     N - 1))
        slope = 0.5 * float(slopes[i1] + slopes[i2])
    intercept = float(np.median(y - slope * x))
    return slope, intercept

# --- MODEL LADEN ----------------------------------------------------------
print("Model laden...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_FILE, compile=False)
print(f"  Model geladen: {MODEL_FILE}")
print(f"  Outputs: {[o.shape for o in model.outputs]}")

# --- LABELS LADEN ---------------------------------------------------------
labels_df = pd.read_csv(LABELS_CSV, sep=";", dtype={"sample_id": str})
labels_df["sample_id"] = labels_df["sample_id"].str.strip()
labels_df.columns = [c.strip() for c in labels_df.columns]

lvh_col = next((c for c in ["lvh_label","LVH_label","LVH","lvh","lvh_status","EILVH","eilvh"]
                if c in labels_df.columns), None)

for col in ["age","sex","bmi","lvm_grams","LVM_dubBSA"]:
    if col in labels_df.columns:
        labels_df[col] = labels_df[col].astype(str).str.replace(",", ".").astype(float)

def parse_lvh(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"1","1.0","ja","yes","y","true","t","lvh","lvh+","positive","pos"}: return 1
    if s in {"0","0.0","nee","no","n","false","f","geen","geen lvh","geen_lvh",
             "no lvh","no_lvh","lvh-","negative","neg","control"}: return 0
    try:
        v = float(s.replace(",", "."))
        if v in (0, 1): return int(v)
    except ValueError: pass
    return np.nan

has_indexed = "LVM_dubBSA" in labels_df.columns
has_lvh     = lvh_col is not None
if has_lvh:
    labels_df["lvh_label_int"] = labels_df[lvh_col].apply(parse_lvh)

print(f"\nLabels geladen: {len(labels_df)} atleten")
print(f"  CMR LVM      : {labels_df['lvm_grams'].mean():.1f} +/- {labels_df['lvm_grams'].std():.1f} g")
if has_indexed:
    print(f"  CMR LVM/BSA  : {labels_df['LVM_dubBSA'].mean():.1f} +/- {labels_df['LVM_dubBSA'].std():.1f} g/m^2")
if has_lvh:
    print(f"  LVH-kolom    : '{lvh_col}', verdeling {labels_df['lvh_label_int'].value_counts(dropna=False).to_dict()}")

# --- XML LOADER -----------------------------------------------------------
def decode_b64(raw, scale=1.0):
    return np.frombuffer(base64.b64decode(raw), dtype="<i2").astype(np.float32) * scale

def load_xml(path):
    import bs4
    with open(path, "r", errors="replace") as f:
        soup = bs4.BeautifulSoup(f, "lxml")
    v = {}
    for wf in soup.find_all("waveform"):
        wt = wf.find("waveformtype")
        if wt is None or wt.text.strip() != "Rhythm": continue
        for ld in wf.find_all("leaddata"):
            lid = ld.find("leadid").text.strip()
            sc_tag = ld.find("leadamplitudeunitsperbit")
            sc = float(sc_tag.text.replace(",", ".")) if sc_tag else 1.0
            v[lid] = decode_b64(ld.find("waveformdata").text.strip(), sc)
        break
    if "III" not in v: v["III"] = v["II"] - v["I"]
    if "aVR" not in v: v["aVR"] = -(v["I"] + v["II"]) / 2
    if "aVL" not in v: v["aVL"] =  v["I"] - v["II"] / 2
    if "aVF" not in v: v["aVF"] =  v["II"] - v["I"] / 2
    def resample(x):
        n = len(x)
        return x if n == ECG_SAMPLES else np.interp(np.linspace(0, n, ECG_SAMPLES), np.arange(n), x)
    return np.column_stack([resample(v[l]) for l in LEAD_ORDER]).astype(np.float32)

# --- INFERENTIE -----------------------------------------------------------
print("\nInferentie starten...")
ecg_index = {Path(p).stem: str(p) for p in Path(ECG_DIR).glob("*.xml")}
print(f"  ECG-bestanden gevonden: {len(ecg_index)}")

n_inputs = len(model.inputs)
records, fouten = [], 0
for i, row in labels_df.iterrows():
    sid = str(row["sample_id"])
    if sid not in ecg_index: fouten += 1; continue
    try:
        ecg  = load_xml(ecg_index[sid])
        norm = (ecg / ECG_NORM)[np.newaxis, ...]
        age_n = np.array([[(float(row["age"]) - AGE_MEAN) / AGE_STD]], dtype=np.float32)
        bmi_n = np.array([[(float(row["bmi"]) - BMI_MEAN) / BMI_STD]], dtype=np.float32)
        sex_val = int(row["sex"])
        sex = np.array([[1 - sex_val, sex_val]], dtype=np.float32)

        if n_inputs == 4:   out = model.predict([norm, age_n, sex, bmi_n], verbose=0)
        elif n_inputs == 2: out = model.predict([norm, np.array([[age_n[0,0], sex_val, bmi_n[0,0]]], dtype=np.float32)], verbose=0)
        else:               out = model.predict(norm, verbose=0)

        reg_raw = None
        if isinstance(out, (list, tuple)):
            for o in out:
                if o.shape[-1] == 1:
                    reg_raw = float(o[0, 0]); break
        else:
            reg_raw = float(out[0, 0])
        if reg_raw is None: fouten += 1; continue

        rec = {
            "sample_id":         sid,
            "age":               float(row["age"]),
            "sex":               sex_val,
            "bmi":               float(row["bmi"]),
            "lvm_cmr":           float(row["lvm_grams"]),
            "lvm_predicted":     reg_raw * LVM_STD + LVM_MEAN,
            "lvm_predicted_raw": reg_raw,
        }
        if has_indexed:
            rec["lvm_cmr_indexed"]       = float(row["LVM_dubBSA"])
            bsa = rec["lvm_cmr"] / rec["lvm_cmr_indexed"] if rec["lvm_cmr_indexed"] else np.nan
            rec["bsa"]                   = bsa
            rec["lvm_predicted_indexed"] = rec["lvm_predicted"] / bsa if bsa else np.nan
        if has_lvh:
            rec["lvh_label_int"] = row.get("lvh_label_int", np.nan)
        records.append(rec)
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(labels_df)} verwerkt...")
    except Exception as e:
        print(f"  FOUT bij {sid}: {e}"); fouten += 1

results = pd.DataFrame(records)
print(f"\nKlaar: {len(results)} voorspellingen, {fouten} fouten")
if len(results) == 0: sys.exit("Geen voorspellingen gelukt.")
results = results[np.isfinite(results["lvm_predicted"]) & np.isfinite(results["lvm_cmr"])].reset_index(drop=True)
if len(results) < 3: sys.exit("Te weinig geldige voorspellingen.")

# --- METRIC HELPERS -------------------------------------------------------
def metrics(pred, truth, label, seed=42):
    pred  = np.asarray(pred, dtype=float); truth = np.asarray(truth, dtype=float)
    mask  = np.isfinite(pred) & np.isfinite(truth); pred, truth = pred[mask], truth[mask]
    if len(pred) < 3:
        return dict(label=label, n=len(pred), rho=np.nan, p=np.nan, rho_lo=np.nan, rho_hi=np.nan,
                    mae=np.nan, m_lo=np.nan, m_hi=np.nan, mean_d=np.nan, sd_d=np.nan,
                    loa_lo=np.nan, loa_hi=np.nan,
                    smape=np.nan, bias_pct=np.nan, loa_pct_lo=np.nan, loa_pct_hi=np.nan)
    rho, p_val = stats.spearmanr(pred, truth)
    diff = truth - pred
    mae = np.mean(np.abs(diff)); mean_d, sd_d = diff.mean(), diff.std()
    mean_v = (pred + truth) / 2
    safe   = np.abs(mean_v) > 1e-9
    diff_pct = 100.0 * (truth[safe] - pred[safe]) / mean_v[safe]
    smape    = float(np.mean(np.abs(diff_pct)))
    bias_pct = float(diff_pct.mean()); sd_pct = float(diff_pct.std())
    rng = np.random.default_rng(seed); boot_rho, boot_m = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(truth), len(truth))
        boot_rho.append(stats.spearmanr(pred[idx], truth[idx])[0])
        boot_m.append(np.mean(np.abs(truth[idx] - pred[idx])))
    rho_lo, rho_hi = np.percentile(boot_rho, [2.5, 97.5])
    m_lo, m_hi     = np.percentile(boot_m,   [2.5, 97.5])
    return dict(label=label, n=len(pred), rho=rho, p=p_val, rho_lo=rho_lo, rho_hi=rho_hi,
                mae=mae, m_lo=m_lo, m_hi=m_hi, mean_d=mean_d, sd_d=sd_d,
                loa_lo=mean_d - 1.96*sd_d, loa_hi=mean_d + 1.96*sd_d,
                smape=smape, bias_pct=bias_pct,
                loa_pct_lo=bias_pct - 1.96*sd_pct, loa_pct_hi=bias_pct + 1.96*sd_pct)

def print_metrics(m):
    print(f"\n  {m['label']} (n={m['n']})")
    if np.isnan(m['rho']): print(f"    (te weinig data)"); return
    print(f"    Spearman rho            : {m['rho']:.3f}  95%CI [{m['rho_lo']:.3f}, {m['rho_hi']:.3f}]   p={m['p']:.2e}")
    print(f"    MAE (absoluut)          : {m['mae']:.1f}  95%CI [{m['m_lo']:.1f}, {m['m_hi']:.1f}]")
    print(f"    sMAPE (relatief)        : {m['smape']:.1f}%")
    print(f"    Bias absoluut (CMR-Pred): {m['mean_d']:+.1f}      LoA [{m['loa_lo']:+.1f}, {m['loa_hi']:+.1f}]")
    print(f"    Bias relatief (CMR-Pred): {m['bias_pct']:+.1f}%   LoA [{m['loa_pct_lo']:+.1f}%, {m['loa_pct_hi']:+.1f}%]")

def _split_by_color(x, y, color_by, color_map):
    cb = np.asarray(color_by); groups = []; plotted = np.zeros(len(cb), dtype=bool)
    for val, col, lbl in color_map:
        sel = cb == val; plotted |= sel
        if sel.any(): groups.append((x[sel], y[sel], col, lbl))
    unk = ~plotted
    if unk.any(): groups.append((x[unk], y[unk], "lightgray", f"Onbekend (n={unk.sum()})"))
    return groups

def scatter_panel(ax, pred, truth, title, xlim, ylim,
                  color_by=None, color_map=None, seed=42):
    pred = np.asarray(pred, dtype=float); truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth); pred, truth = pred[mask], truth[mask]
    if color_by is not None and color_map is not None:
        cb = np.asarray(color_by)[mask]
        for x_, y_, col, lbl in _split_by_color(pred, truth, cb, color_map):
            ax.scatter(x_, y_, alpha=0.55, s=20, color=col, label=lbl)
    else:
        ax.scatter(pred, truth, alpha=0.4, s=20, color="royalblue")
    lo_lim = min(xlim[0], ylim[0]); hi_lim = max(xlim[1], ylim[1])
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "--", color="gray", label="Identity")
    if len(pred) >= 3:
        rho_val, _ = stats.spearmanr(pred, truth)
        mae_val = float(np.mean(np.abs(truth - pred)))
        rng = np.random.default_rng(seed); boot_rho, boot_mae = [], []
        for _ in range(1000):
            idx = rng.integers(0, len(truth), len(truth))
            boot_rho.append(stats.spearmanr(pred[idx], truth[idx])[0])
            boot_mae.append(np.mean(np.abs(truth[idx] - pred[idx])))
        rho_lo, rho_hi = np.percentile(boot_rho, [2.5, 97.5])
        mae_lo, mae_hi = np.percentile(boot_mae, [2.5, 97.5])
        ax.set_title(
            f"{title}\n"
            f"rho={rho_val:.2f} (95% CI {rho_lo:.2f}-{rho_hi:.2f}), "
            f"MAE={mae_val:.1f} (95% CI {mae_lo:.1f}-{mae_hi:.1f})",
            fontsize=10
        )
    else:
        ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("CMR-derived")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc=LEGEND_LOC)

def ba_panel(ax, pred, truth, title, xlim_ba, ylim_ba, color_by=None, color_map=None):
    pred = np.asarray(pred, dtype=float); truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth); pred, truth = pred[mask], truth[mask]
    if len(pred) == 0:
        ax.set_title(title); ax.text(0.5, 0.5, "Geen data", ha="center", transform=ax.transAxes); return
    mean_v = (pred + truth) / 2
    diff_v = truth - pred
    md, sd = diff_v.mean(), diff_v.std()
    if color_by is not None and color_map is not None:
        cb = np.asarray(color_by)[mask]
        for x_, y_, col, lbl in _split_by_color(mean_v, diff_v, cb, color_map):
            ax.scatter(x_, y_, alpha=0.55, s=20, color=col, label=lbl)
    else:
        ax.scatter(mean_v, diff_v, alpha=0.4, s=20, color="royalblue")
    ax.axhline(y=md,           color="red", linestyle="-",  label=f"Bias {md:+.1f}")
    ax.axhline(y=md + 1.96*sd, color="red", linestyle="--", label=f"+1.96 SD {md+1.96*sd:+.1f}")
    ax.axhline(y=md - 1.96*sd, color="red", linestyle="--", label=f"-1.96 SD {md-1.96*sd:+.1f}")
    ax.axhline(y=0, color="gray", linestyle=":")
    ax.set_xlabel("Mean of CMR and predicted")
    ax.set_ylabel("CMR - Predicted")
    ax.set_title(f"BA absoluut: {title}")
    ax.set_xlim(xlim_ba); ax.set_ylim(ylim_ba)
    ax.legend(fontsize=8, loc=LEGEND_LOC)

def ba_panel_relative(ax, pred, truth, title, xlim_ba, ylim_ba_pct,
                      color_by=None, color_map=None):
    pred = np.asarray(pred, dtype=float); truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth); pred, truth = pred[mask], truth[mask]
    if color_by is not None: color_by = np.asarray(color_by)[mask]
    if len(pred) == 0:
        ax.set_title(title); ax.text(0.5, 0.5, "Geen data", ha="center", transform=ax.transAxes); return
    mean_v = (pred + truth) / 2
    safe   = np.abs(mean_v) > 1e-9
    mean_v = mean_v[safe]; pred = pred[safe]; truth = truth[safe]
    if color_by is not None: color_by = color_by[safe]
    diff_pct = 100.0 * (truth - pred) / mean_v
    md, sd = diff_pct.mean(), diff_pct.std()
    if color_by is not None and color_map is not None:
        for x_, y_, col, lbl in _split_by_color(mean_v, diff_pct, color_by, color_map):
            ax.scatter(x_, y_, alpha=0.55, s=20, color=col, label=lbl)
    else:
        ax.scatter(mean_v, diff_pct, alpha=0.4, s=20, color="royalblue")
    ax.axhline(y=md,           color="red", linestyle="-",  label=f"Bias {md:+.1f}%")
    ax.axhline(y=md + 1.96*sd, color="red", linestyle="--", label=f"+1.96 SD {md+1.96*sd:+.1f}%")
    ax.axhline(y=md - 1.96*sd, color="red", linestyle="--", label=f"-1.96 SD {md-1.96*sd:+.1f}%")
    ax.axhline(y=0, color="gray", linestyle=":")
    ax.set_xlabel("Mean of CMR and predicted")
    ax.set_ylabel("(CMR - Predicted) / Mean * 100 (%)")
    ax.set_title(f"BA relatief: {title}")
    ax.set_xlim(xlim_ba); ax.set_ylim(ylim_ba_pct)
    ax.legend(fontsize=8, loc=LEGEND_LOC)

# --- 1. ABSOLUTE LVM - OLS-kalibratie ------------------------------------
lvm_cmr  = results["lvm_cmr"].values
lvm_pred = results["lvm_predicted"].values
lvh_arr  = results["lvh_label_int"].values if has_lvh else None
cmap     = LVH_COLOR_MAP if has_lvh else None

calib_abs = LinearRegression().fit(lvm_pred.reshape(-1, 1), lvm_cmr)
slope_abs, intercept_abs = float(calib_abs.coef_[0]), float(calib_abs.intercept_)
lvm_pred_cal = calib_abs.predict(lvm_pred.reshape(-1, 1))
results["lvm_predicted_calibrated"] = lvm_pred_cal

pb_slope_abs, pb_intercept_abs = passing_bablok(lvm_pred, lvm_cmr)
lvm_pred_pb = pb_slope_abs * lvm_pred + pb_intercept_abs
results["lvm_predicted_pb"] = lvm_pred_pb

# Gedeelde limieten - gebruikt in fig1, fig3 (abs) en fig4 (abs-rij)
sc_lim1   = data_lims(lvm_cmr, lvm_pred, lvm_pred_cal, lvm_pred_pb)
ba_x_lim1 = data_lims((lvm_pred+lvm_cmr)/2, (lvm_pred_cal+lvm_cmr)/2, (lvm_pred_pb+lvm_cmr)/2)
ba_y_lim1 = data_lims(lvm_cmr - lvm_pred, lvm_cmr - lvm_pred_cal, lvm_cmr - lvm_pred_pb)
ba_y_lim1_pct = data_lims(pct_diffs(lvm_pred, lvm_cmr),
                          pct_diffs(lvm_pred_cal, lvm_cmr),
                          pct_diffs(lvm_pred_pb,  lvm_cmr))

print(f"\n{'='*64}")
print(f"  1. ABSOLUTE LVM (g) - primary: OLS  (bias = CMR - Predicted)")
print(f"{'='*64}")
print(f"  Gedeelde scatter-as    : {sc_lim1[0]:.0f} - {sc_lim1[1]:.0f} g")
print(f"  Gedeelde BA-x-as       : {ba_x_lim1[0]:.0f} - {ba_x_lim1[1]:.0f} g")
print(f"  Gedeelde BA-y-as abs   : {ba_y_lim1[0]:.0f} - {ba_y_lim1[1]:.0f} g")
print(f"  Gedeelde BA-y-as %     : {ba_y_lim1_pct[0]:.1f} - {ba_y_lim1_pct[1]:.1f} %")
print(f"  CMR LVM                : {np.mean(lvm_cmr):.1f} +/- {np.std(lvm_cmr):.1f}  (range {lvm_cmr.min():.0f}-{lvm_cmr.max():.0f})")
print(f"  Predicted LVM (ruw)    : {np.mean(lvm_pred):.1f} +/- {np.std(lvm_pred):.1f}  (range {lvm_pred.min():.0f}-{lvm_pred.max():.0f})")
print(f"  OLS-formule            : CMR = {slope_abs:.3f} * predicted + {intercept_abs:+.1f}")
print(f"  PB-formule  (sensit.)  : CMR = {pb_slope_abs:.3f} * predicted + {pb_intercept_abs:+.1f}")
print_metrics(metrics(lvm_pred,      lvm_cmr, "Ruw"))
print_metrics(metrics(lvm_pred_cal,  lvm_cmr, "OLS-gerecalibreerd"))
print_metrics(metrics(lvm_pred_pb,   lvm_cmr, "PB-gerecalibreerd (sensitivity)"))

fig1, ax1 = plt.subplots(2, 3, figsize=(18, 11))
scatter_panel(ax1[0,0], lvm_pred, lvm_cmr, "Ruw",
              sc_lim1, sc_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel(ax1[0,1], lvm_pred, lvm_cmr, "Ruw",
         ba_x_lim1, ba_y_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel_relative(ax1[0,2], lvm_pred, lvm_cmr, "Ruw",
                  ba_x_lim1, ba_y_lim1_pct, color_by=lvh_arr, color_map=cmap)
scatter_panel(ax1[1,0], lvm_pred_cal, lvm_cmr, "OLS-gerecalibreerd",
              sc_lim1, sc_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel(ax1[1,1], lvm_pred_cal, lvm_cmr, "OLS-gerecalibreerd",
         ba_x_lim1, ba_y_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel_relative(ax1[1,2], lvm_pred_cal, lvm_cmr, "OLS-gerecalibreerd",
                  ba_x_lim1, ba_y_lim1_pct, color_by=lvh_arr, color_map=cmap)
fig1.suptitle("LVM-AI vs CMR - Absolute LVM (g), OLS-kalibratie  (CMR - Predicted)", fontsize=14)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_absoluut.png"), dpi=150)
print(f"\n  Figuur 1 opgeslagen: {OUTPUT_DIR}/fig1_absoluut.png")

# --- 2. GEINDEXEERDE LVM - OLS-kalibratie --------------------------------
sc_lim2 = ba_x_lim2 = ba_y_lim2 = ba_y_lim2_pct = None
if has_indexed:
    ix_mask = np.isfinite(results["lvm_predicted_indexed"]) & np.isfinite(results["lvm_cmr_indexed"])
    if ix_mask.sum() >= 3:
        lvm_cmr_ix  = results.loc[ix_mask, "lvm_cmr_indexed"].values
        lvm_pred_ix = results.loc[ix_mask, "lvm_predicted_indexed"].values
        lvh_arr_ix  = results.loc[ix_mask, "lvh_label_int"].values if has_lvh else None

        calib_ix = LinearRegression().fit(lvm_pred_ix.reshape(-1, 1), lvm_cmr_ix)
        slope_ix, intercept_ix = float(calib_ix.coef_[0]), float(calib_ix.intercept_)
        lvm_pred_ix_cal = calib_ix.predict(lvm_pred_ix.reshape(-1, 1))
        results.loc[ix_mask, "lvm_predicted_indexed_calibrated"] = lvm_pred_ix_cal

        pb_slope_ix, pb_intercept_ix = passing_bablok(lvm_pred_ix, lvm_cmr_ix)
        lvm_pred_ix_pb = pb_slope_ix * lvm_pred_ix + pb_intercept_ix
        results.loc[ix_mask, "lvm_predicted_indexed_pb"] = lvm_pred_ix_pb

        sc_lim2   = data_lims(lvm_cmr_ix, lvm_pred_ix, lvm_pred_ix_cal, lvm_pred_ix_pb)
        ba_x_lim2 = data_lims((lvm_pred_ix+lvm_cmr_ix)/2, (lvm_pred_ix_cal+lvm_cmr_ix)/2,
                              (lvm_pred_ix_pb+lvm_cmr_ix)/2)
        ba_y_lim2 = data_lims(lvm_cmr_ix - lvm_pred_ix, lvm_cmr_ix - lvm_pred_ix_cal,
                              lvm_cmr_ix - lvm_pred_ix_pb)
        ba_y_lim2_pct = data_lims(pct_diffs(lvm_pred_ix,     lvm_cmr_ix),
                                  pct_diffs(lvm_pred_ix_cal, lvm_cmr_ix),
                                  pct_diffs(lvm_pred_ix_pb,  lvm_cmr_ix))

        print(f"\n{'='*64}")
        print(f"  2. GEINDEXEERDE LVM (g/m^2) - primary: OLS  (bias = CMR - Predicted)")
        print(f"{'='*64}")
        print(f"  Gedeelde scatter-as    : {sc_lim2[0]:.0f} - {sc_lim2[1]:.0f} g/m^2")
        print(f"  Gedeelde BA-x-as       : {ba_x_lim2[0]:.0f} - {ba_x_lim2[1]:.0f} g/m^2")
        print(f"  Gedeelde BA-y-as abs   : {ba_y_lim2[0]:.0f} - {ba_y_lim2[1]:.0f} g/m^2")
        print(f"  Gedeelde BA-y-as %     : {ba_y_lim2_pct[0]:.1f} - {ba_y_lim2_pct[1]:.1f} %")
        print(f"  CMR LVM/BSA            : {np.mean(lvm_cmr_ix):.1f} +/- {np.std(lvm_cmr_ix):.1f}")
        print(f"  Predicted LVM/BSA (ruw): {np.mean(lvm_pred_ix):.1f} +/- {np.std(lvm_pred_ix):.1f}")
        print(f"  OLS-formule            : CMR/BSA = {slope_ix:.3f} * predicted/BSA + {intercept_ix:+.1f}")
        print(f"  PB-formule  (sensit.)  : CMR/BSA = {pb_slope_ix:.3f} * predicted/BSA + {pb_intercept_ix:+.1f}")
        print_metrics(metrics(lvm_pred_ix,     lvm_cmr_ix, "Geindexeerd ruw"))
        print_metrics(metrics(lvm_pred_ix_cal, lvm_cmr_ix, "Geindexeerd OLS-gerecalibreerd"))
        print_metrics(metrics(lvm_pred_ix_pb,  lvm_cmr_ix, "Geindexeerd PB-gerecalibreerd"))

        fig2, ax2 = plt.subplots(2, 3, figsize=(18, 11))
        scatter_panel(ax2[0,0], lvm_pred_ix, lvm_cmr_ix, "Geindexeerd ruw",
                      sc_lim2, sc_lim2, color_by=lvh_arr_ix, color_map=cmap)
        ba_panel(ax2[0,1], lvm_pred_ix, lvm_cmr_ix, "Geindexeerd ruw",
                 ba_x_lim2, ba_y_lim2, color_by=lvh_arr_ix, color_map=cmap)
        ba_panel_relative(ax2[0,2], lvm_pred_ix, lvm_cmr_ix, "Geindexeerd ruw",
                          ba_x_lim2, ba_y_lim2_pct, color_by=lvh_arr_ix, color_map=cmap)
        scatter_panel(ax2[1,0], lvm_pred_ix_cal, lvm_cmr_ix, "Geindexeerd OLS-gerecalibreerd",
                      sc_lim2, sc_lim2, color_by=lvh_arr_ix, color_map=cmap)
        ba_panel(ax2[1,1], lvm_pred_ix_cal, lvm_cmr_ix, "Geindexeerd OLS-gerecalibreerd",
                 ba_x_lim2, ba_y_lim2, color_by=lvh_arr_ix, color_map=cmap)
        ba_panel_relative(ax2[1,2], lvm_pred_ix_cal, lvm_cmr_ix, "Geindexeerd OLS-gerecalibreerd",
                          ba_x_lim2, ba_y_lim2_pct, color_by=lvh_arr_ix, color_map=cmap)
        fig2.suptitle("LVM-AI vs CMR - Geindexeerde LVM (g/m^2), OLS-kalibratie  (CMR - Predicted)", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_indexed.png"), dpi=150)
        print(f"  Figuur 2 opgeslagen: {OUTPUT_DIR}/fig2_indexed.png")

# --- 3. SUBGROEP LVH+ vs LVH- --------------------------------------------
print(f"\n{'='*64}")
print(f"  3. SUBGROEP: LVH+ vs LVH-")
print(f"{'='*64}")
if not has_lvh:
    print("  Geen LVH-kolom - figuur 3 overgeslagen")
else:
    sub = results.dropna(subset=["lvh_label_int"]).copy()
    sub["lvh_label_int"] = sub["lvh_label_int"].astype(int)
    print(f"  n met geldig label   : {len(sub)} / {len(results)}")
    print(f"  Verdeling            : {sub['lvh_label_int'].value_counts().to_dict()}")

    if sub["lvh_label_int"].nunique() < 2 or len(sub) < 6:
        print("  Te weinig data of slechts een klasse - figuur 3 overgeslagen")
    else:
        g_pos = sub[sub["lvh_label_int"] == 1]
        g_neg = sub[sub["lvh_label_int"] == 0]
        print(f"  LVH+ : n={len(g_pos)},  CMR LVM = {g_pos['lvm_cmr'].mean():.1f} +/- {g_pos['lvm_cmr'].std():.1f} g")
        print(f"  LVH- : n={len(g_neg)},  CMR LVM = {g_neg['lvm_cmr'].mean():.1f} +/- {g_neg['lvm_cmr'].std():.1f} g")

        for grp, df_ in [("LVH+", g_pos), ("LVH-", g_neg)]:
            print(f"\n  -- {grp} --")
            print_metrics(metrics(df_["lvm_predicted"].values,            df_["lvm_cmr"].values, "Absoluut ruw"))
            print_metrics(metrics(df_["lvm_predicted_calibrated"].values, df_["lvm_cmr"].values, "Absoluut OLS-gerecalibreerd"))
            if has_indexed and "lvm_predicted_indexed" in df_.columns:
                print_metrics(metrics(df_["lvm_predicted_indexed"].values, df_["lvm_cmr_indexed"].values, "Geindexeerd ruw"))
            if has_indexed and "lvm_predicted_indexed_calibrated" in df_.columns:
                print_metrics(metrics(df_["lvm_predicted_indexed_calibrated"].values,
                                      df_["lvm_cmr_indexed"].values, "Geindexeerd OLS-gerecalibreerd"))

        bias_pos = (g_pos["lvm_cmr"] - g_pos["lvm_predicted"]).values
        bias_neg = (g_neg["lvm_cmr"] - g_neg["lvm_predicted"]).values
        if len(bias_pos) > 1 and len(bias_neg) > 1:
            u, u_p = stats.mannwhitneyu(bias_pos, bias_neg, alternative="two-sided")
            print(f"\n  Mann-Whitney bias LVH+ vs LVH- (CMR - Pred)  U={u:.0f}, p={u_p:.3g}")

        color_by3 = sub["lvh_label_int"].values
        fig3, ax3 = plt.subplots(2, 4, figsize=(22, 10))

        # Boven-rij scatters: dezelfde limieten als fig1 / fig2
        scatter_panel(ax3[0,0], sub["lvm_predicted"].values, sub["lvm_cmr"].values,
                      "Absoluut ruw", sc_lim1, sc_lim1, color_by=color_by3, color_map=LVH_COLOR_MAP)
        scatter_panel(ax3[0,1], sub["lvm_predicted_calibrated"].values, sub["lvm_cmr"].values,
                      "Absoluut OLS-gerecalibreerd", sc_lim1, sc_lim1, color_by=color_by3, color_map=LVH_COLOR_MAP)
        if has_indexed and "lvm_predicted_indexed" in sub.columns and sc_lim2 is not None:
            scatter_panel(ax3[0,2], sub["lvm_predicted_indexed"].values, sub["lvm_cmr_indexed"].values,
                          "Geindexeerd ruw", sc_lim2, sc_lim2, color_by=color_by3, color_map=LVH_COLOR_MAP)
        else:
            ax3[0,2].axis("off")
        if has_indexed and "lvm_predicted_indexed_calibrated" in sub.columns and sc_lim2 is not None:
            scatter_panel(ax3[0,3], sub["lvm_predicted_indexed_calibrated"].values, sub["lvm_cmr_indexed"].values,
                          "Geindexeerd OLS-gerecalibreerd", sc_lim2, sc_lim2, color_by=color_by3, color_map=LVH_COLOR_MAP)
        else:
            ax3[0,3].axis("off")

        cols = ["steelblue", "crimson"]
        boxplot_specs = [
            (ax3[1,0], [g_neg["lvm_cmr"].values, g_pos["lvm_cmr"].values],
             "CMR LV massa per groep", "LVM (g)", "abs"),
            (ax3[1,1], [g_neg["lvm_predicted"].values, g_pos["lvm_predicted"].values],
             "Predicted absoluut (ruw)", "Predicted LVM (g)", "abs"),
            (ax3[1,2], [g_neg["lvm_predicted_calibrated"].values, g_pos["lvm_predicted_calibrated"].values],
             "Predicted absoluut (OLS-cal)", "Predicted LVM (g)", "abs"),
        ]
        if (has_indexed and "lvm_predicted_indexed_calibrated" in sub.columns
                and g_neg["lvm_predicted_indexed_calibrated"].notna().any()
                and g_pos["lvm_predicted_indexed_calibrated"].notna().any()):
            boxplot_specs.append(
                (ax3[1,3],
                 [g_neg["lvm_predicted_indexed_calibrated"].dropna().values,
                  g_pos["lvm_predicted_indexed_calibrated"].dropna().values],
                 "Predicted geindexeerd (OLS-cal)", "Predicted LVM (g/m^2)", "ix")
            )
        else:
            ax3[1,3].axis("off")

        # Gedeelde y-limieten boxplots == scatter-limieten (fig1 / fig2)
        box_lim_abs = sc_lim1
        box_lim_ix  = sc_lim2 if sc_lim2 is not None else (0, 1)

        for ax_, data, title, ylab, unit in boxplot_specs:
            bp = boxplot_compat(ax_, data, ["LVH-", "LVH+"], patch_artist=True)
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col); patch.set_alpha(0.6)
            ax_.set_title(title); ax_.set_ylabel(ylab)
            ax_.set_ylim(box_lim_abs if unit == "abs" else box_lim_ix)
            if len(data[0]) > 1 and len(data[1]) > 1:
                _, p_mw = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
                ax_.text(0.05, 0.95, f"Mann-Whitney p={p_mw:.3g}",
                         transform=ax_.transAxes, va="top", fontsize=9)

        fig3.suptitle("Subgroep: LVH+ vs LVH- (Spearman)", fontsize=14)
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_lvh_subgroep.png"), dpi=150)
        print(f"\n  Figuur 3 opgeslagen: {OUTPUT_DIR}/fig3_lvh_subgroep.png")

# --- 4. PASSING-BABLOK SENSITIVITY ---------------------------------------
print(f"\n{'='*64}")
print(f"  4. SENSITIVITY: Passing-Bablok kalibratie  (CMR - Predicted)")
print(f"{'='*64}")
print(f"  Doel: robuustheid van OLS-kalibratie toetsen tegen niet-parametrische PB.")
print(f"  Absoluut    : CMR = {pb_slope_abs:.3f} * predicted + {pb_intercept_abs:+.1f}")
if has_indexed and 'pb_slope_ix' in dir():
    print(f"  Geindexeerd : CMR/BSA = {pb_slope_ix:.3f} * predicted/BSA + {pb_intercept_ix:+.1f}")

print(f"\n  -- Totaal (PB-gerecalibreerd) --")
print_metrics(metrics(lvm_pred_pb, lvm_cmr, "Absoluut PB"))
if has_indexed and "lvm_predicted_indexed_pb" in results.columns:
    ix_mask_pb = np.isfinite(results["lvm_predicted_indexed_pb"]) & np.isfinite(results["lvm_cmr_indexed"])
    print_metrics(metrics(results.loc[ix_mask_pb, "lvm_predicted_indexed_pb"].values,
                          results.loc[ix_mask_pb, "lvm_cmr_indexed"].values,
                          "Geindexeerd PB"))

if has_lvh:
    sub_pb = results.dropna(subset=["lvh_label_int"]).copy()
    sub_pb["lvh_label_int"] = sub_pb["lvh_label_int"].astype(int)
    if sub_pb["lvh_label_int"].nunique() >= 2 and len(sub_pb) >= 6:
        g_pos_pb = sub_pb[sub_pb["lvh_label_int"] == 1]
        g_neg_pb = sub_pb[sub_pb["lvh_label_int"] == 0]

        for grp, df_ in [("LVH+", g_pos_pb), ("LVH-", g_neg_pb)]:
            print(f"\n  -- {grp} (PB-gerecalibreerd) --")
            print_metrics(metrics(df_["lvm_predicted_pb"].values, df_["lvm_cmr"].values, "Absoluut PB"))
            if has_indexed and "lvm_predicted_indexed_pb" in df_.columns:
                mask_ix = df_["lvm_predicted_indexed_pb"].notna() & df_["lvm_cmr_indexed"].notna()
                if mask_ix.sum() >= 3:
                    print_metrics(metrics(df_.loc[mask_ix, "lvm_predicted_indexed_pb"].values,
                                          df_.loc[mask_ix, "lvm_cmr_indexed"].values,
                                          "Geindexeerd PB"))

        bias_pos_pb = (g_pos_pb["lvm_cmr"] - g_pos_pb["lvm_predicted_pb"]).values
        bias_neg_pb = (g_neg_pb["lvm_cmr"] - g_neg_pb["lvm_predicted_pb"]).values
        if len(bias_pos_pb) > 1 and len(bias_neg_pb) > 1:
            u_pb, u_p_pb = stats.mannwhitneyu(bias_pos_pb, bias_neg_pb, alternative="two-sided")
            print(f"\n  Mann-Whitney PB-bias LVH+ vs LVH- (CMR - Pred)  U={u_pb:.0f}, p={u_p_pb:.3g}")

        if has_indexed and "lvm_predicted_indexed_pb" in g_pos_pb.columns:
            ix_pos = g_pos_pb.dropna(subset=["lvm_predicted_indexed_pb", "lvm_cmr_indexed"])
            ix_neg = g_neg_pb.dropna(subset=["lvm_predicted_indexed_pb", "lvm_cmr_indexed"])
            if len(ix_pos) > 1 and len(ix_neg) > 1:
                bias_pos_ix_pb = (ix_pos["lvm_cmr_indexed"] - ix_pos["lvm_predicted_indexed_pb"]).values
                bias_neg_ix_pb = (ix_neg["lvm_cmr_indexed"] - ix_neg["lvm_predicted_indexed_pb"]).values
                u_ix_pb, u_p_ix_pb = stats.mannwhitneyu(bias_pos_ix_pb, bias_neg_ix_pb, alternative="two-sided")
                print(f"  Mann-Whitney PB-bias (geindexeerd) LVH+ vs LVH-  U={u_ix_pb:.0f}, p={u_p_ix_pb:.3g}")

print(f"\n  -- OLS vs PB vergelijking (totaal) --")
m_ols_abs = metrics(lvm_pred_cal, lvm_cmr, "OLS abs")
m_pb_abs  = metrics(lvm_pred_pb,  lvm_cmr, "PB abs")
print(f"  {'Metric':<20}{'OLS':>15}{'PB':>15}{'Delta(PB-OLS)':>15}")
print(f"  {'-'*65}")
print(f"  {'Spearman rho':<20}{m_ols_abs['rho']:>15.3f}{m_pb_abs['rho']:>15.3f}{m_pb_abs['rho']-m_ols_abs['rho']:>+15.3f}")
print(f"  {'MAE (g)':<20}{m_ols_abs['mae']:>15.1f}{m_pb_abs['mae']:>15.1f}{m_pb_abs['mae']-m_ols_abs['mae']:>+15.1f}")
print(f"  {'sMAPE (%)':<20}{m_ols_abs['smape']:>15.1f}{m_pb_abs['smape']:>15.1f}{m_pb_abs['smape']-m_ols_abs['smape']:>+15.1f}")
print(f"  {'Bias (g)':<20}{m_ols_abs['mean_d']:>+15.1f}{m_pb_abs['mean_d']:>+15.1f}{m_pb_abs['mean_d']-m_ols_abs['mean_d']:>+15.1f}")
print(f"  {'LoA breedte (g)':<20}{m_ols_abs['loa_hi']-m_ols_abs['loa_lo']:>15.1f}{m_pb_abs['loa_hi']-m_pb_abs['loa_lo']:>15.1f}{(m_pb_abs['loa_hi']-m_pb_abs['loa_lo'])-(m_ols_abs['loa_hi']-m_ols_abs['loa_lo']):>+15.1f}")

if has_indexed and "lvm_predicted_indexed_pb" in results.columns:
    ix_mask_cmp = (np.isfinite(results["lvm_predicted_indexed_pb"]) &
                   np.isfinite(results["lvm_cmr_indexed"]) &
                   np.isfinite(results["lvm_predicted_indexed_calibrated"]))
    if ix_mask_cmp.sum() >= 3:
        m_ols_ix = metrics(results.loc[ix_mask_cmp, "lvm_predicted_indexed_calibrated"].values,
                           results.loc[ix_mask_cmp, "lvm_cmr_indexed"].values, "OLS ix")
        m_pb_ix  = metrics(results.loc[ix_mask_cmp, "lvm_predicted_indexed_pb"].values,
                           results.loc[ix_mask_cmp, "lvm_cmr_indexed"].values, "PB ix")
        print(f"\n  -- OLS vs PB vergelijking (geindexeerd) --")
        print(f"  {'Metric':<20}{'OLS':>15}{'PB':>15}{'Delta(PB-OLS)':>15}")
        print(f"  {'-'*65}")
        print(f"  {'Spearman rho':<20}{m_ols_ix['rho']:>15.3f}{m_pb_ix['rho']:>15.3f}{m_pb_ix['rho']-m_ols_ix['rho']:>+15.3f}")
        print(f"  {'MAE (g/m^2)':<20}{m_ols_ix['mae']:>15.1f}{m_pb_ix['mae']:>15.1f}{m_pb_ix['mae']-m_ols_ix['mae']:>+15.1f}")
        print(f"  {'sMAPE (%)':<20}{m_ols_ix['smape']:>15.1f}{m_pb_ix['smape']:>15.1f}{m_pb_ix['smape']-m_ols_ix['smape']:>+15.1f}")
        print(f"  {'Bias (g/m^2)':<20}{m_ols_ix['mean_d']:>+15.1f}{m_pb_ix['mean_d']:>+15.1f}{m_pb_ix['mean_d']-m_ols_ix['mean_d']:>+15.1f}")
        print(f"  {'LoA breedte (g/m^2)':<20}{m_ols_ix['loa_hi']-m_ols_ix['loa_lo']:>15.1f}{m_pb_ix['loa_hi']-m_pb_ix['loa_lo']:>15.1f}{(m_pb_ix['loa_hi']-m_pb_ix['loa_lo'])-(m_ols_ix['loa_hi']-m_ols_ix['loa_lo']):>+15.1f}")

fig4, ax4 = plt.subplots(2, 3, figsize=(18, 11))
# Rij 1: absoluut PB - dezelfde limieten als fig1
scatter_panel(ax4[0,0], lvm_pred_pb, lvm_cmr, "Absoluut PB-gerecalibreerd",
              sc_lim1, sc_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel(ax4[0,1], lvm_pred_pb, lvm_cmr, "Absoluut PB-gerecalibreerd",
         ba_x_lim1, ba_y_lim1, color_by=lvh_arr, color_map=cmap)
ba_panel_relative(ax4[0,2], lvm_pred_pb, lvm_cmr, "Absoluut PB-gerecalibreerd",
                  ba_x_lim1, ba_y_lim1_pct, color_by=lvh_arr, color_map=cmap)

# Rij 2: geindexeerd PB - dezelfde limieten als fig2
if has_indexed and "lvm_predicted_indexed_pb" in results.columns and sc_lim2 is not None:
    ix_mask2 = np.isfinite(results["lvm_predicted_indexed_pb"]) & np.isfinite(results["lvm_cmr_indexed"])
    pb_ix_vals  = results.loc[ix_mask2, "lvm_predicted_indexed_pb"].values
    cmr_ix_vals = results.loc[ix_mask2, "lvm_cmr_indexed"].values
    lvh_ix_vals = results.loc[ix_mask2, "lvh_label_int"].values if has_lvh else None
    scatter_panel(ax4[1,0], pb_ix_vals, cmr_ix_vals, "Geindexeerd PB-gerecalibreerd",
                  sc_lim2, sc_lim2, color_by=lvh_ix_vals, color_map=cmap)
    ba_panel(ax4[1,1], pb_ix_vals, cmr_ix_vals, "Geindexeerd PB-gerecalibreerd",
             ba_x_lim2, ba_y_lim2, color_by=lvh_ix_vals, color_map=cmap)
    ba_panel_relative(ax4[1,2], pb_ix_vals, cmr_ix_vals, "Geindexeerd PB-gerecalibreerd",
                      ba_x_lim2, ba_y_lim2_pct, color_by=lvh_ix_vals, color_map=cmap)
else:
    for k in range(3): ax4[1,k].axis("off")

fig4.suptitle("Sensitivity analysis - Passing-Bablok kalibratie  (CMR - Predicted)", fontsize=14)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_passing_bablok.png"), dpi=150)
print(f"\n  Figuur 4 opgeslagen: {OUTPUT_DIR}/fig4_passing_bablok.png")

output_csv = os.path.join(OUTPUT_DIR, "regressie_voorspellingen.csv")
results.to_csv(output_csv, index=False)
print(f"\nVoorspellingen opgeslagen: {output_csv}")
print(f"Klaar.")

# --- Alle figuren in een keer tonen --------------------------------------
plt.show()
