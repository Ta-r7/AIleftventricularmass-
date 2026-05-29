"""
LVM-AI Saliency Mapping - Externe validatie (Athleten-cohort)
- Identieke methode als origineel ml4h (Khurshid et al.):
  * Vanilla gradients: dOutput / dInput_ECG
  * RMS-normalisatie: grads /= sqrt(mean(grads^2)) + 1e-6
  * Gaussian blur sigma=5.0
  * RGB-overlay: rood=negatief, groen=positief, blauw=ruw signaal
  * 3x4 lead-grid layout
- Toegevoegd voor domain-shift evaluatie:
  * Mean saliency LVH+ vs LVH-
  * Per-lead importance (mean |saliency|)
  * Beide model-heads: regressie (LVM) en classificatie (P(LVH))
Gebruik: python saliency_external_validation.py
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
    "numpy", "pandas", "scipy",
    "matplotlib",
])

import os
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# --- CONFIGURATIE (identiek aan validate_regression.py) -------------------
MODEL_FILE = "ecg_rest_raw_age_sex_bmi_lvm_asymmetric_loss.h5"
ECG_DIR    = "ecg_bestanden"
LABELS_CSV = "regressionlabels.csv"
OUTPUT_DIR = "resultaten_saliency"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ECG_NORM = 2000.0
AGE_MEAN, AGE_STD = 63.35798891483556, 7.554638350423902
BMI_MEAN, BMI_STD = 27.3397, 4.7721
LVM_MEAN, LVM_STD = 89.70372484725051, 24.803669503436304
LEAD_ORDER  = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
ECG_SAMPLES = 5000
SAMPLE_RATE = 500  # Hz

# Saliency-specifieke parameters (identiek aan ml4h)
BLUR_SIGMA   = 5.0
RMS_EPS      = 1e-6
N_INDIV_PLOTS = 6  # aantal individuele plots per groep om op te slaan

# --- XML LOADER (identiek aan validate_regression.py) ---------------------
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

# --- SALIENCY CORE (mirrors ml4h/models/inspect.py:_gradients_from_output) ---
def compute_saliency(model, ecg_in, age_in, sex_in, bmi_in, head_index, output_index=0):
    """
    Vanilla gradient: dOutput / dInput_ECG.
    Identieke RMS-normalisatie als ml4h:
        grads /= sqrt(mean(grads^2)) + 1e-6
    head_index: 0 = regressie (LVM), 1 = classificatie (P(LVH))
    output_index: voor classificatie 1 = P(LVH=1)
    """
    import tensorflow as tf
    ecg_tf = tf.convert_to_tensor(ecg_in)
    with tf.GradientTape() as tape:
        tape.watch(ecg_tf)
        outputs = model([ecg_tf, age_in, sex_in, bmi_in], training=False)
        if isinstance(outputs, (list, tuple)):
            target = outputs[head_index][:, output_index]
        else:
            target = outputs[:, output_index]
    grads = tape.gradient(target, ecg_tf).numpy()
    grads /= (np.sqrt(np.mean(grads**2)) + RMS_EPS)
    return grads[0]  # (5000, 12)

# --- POST-PROCESSING (mirrors ml4h/plots.py:_saliency_blurred_and_scaled) ---
def saliency_blurred_and_scaled(gradients, blur_radius=BLUR_SIGMA, max_value=1.0):
    blurred = gaussian_filter(gradients, sigma=blur_radius)
    mn, mx = blurred.min(), blurred.max()
    if mx - mn > 1e-12:
        blurred = (blurred - mn) / (mx - mn) * max_value
    blurred -= blurred.mean()
    return blurred

# --- PLOTTING (3x4 grid per lead, identiek aan ml4h-stijl) ----------------
def plot_saliency_per_lead(ecg, gradients, title, save_path):
    """ECG (zwart) + saliency (rood) per lead, 3x4 grid."""
    blurred = saliency_blurred_and_scaled(gradients, BLUR_SIGMA)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True)
    t = np.arange(ECG_SAMPLES) / SAMPLE_RATE * 1000.0  # ms
    for k, lead in enumerate(LEAD_ORDER):
        ax = axes[k // 4, k % 4]
        ax.plot(t, ecg[:, k], color="black", lw=0.7, label="ECG")
        ax.set_ylabel("Voltage (norm.)", color="black", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(t, blurred[:, k], color="crimson", lw=0.7, alpha=0.85, label="Saliency")
        ax2.set_ylabel("Saliency", color="crimson", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=7)
        ax.set_title(f"Lead {lead}", fontsize=10)
        if k // 4 == 2:
            ax.set_xlabel("Time (ms)", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_per_lead_importance(per_lead_mean, title, save_path, lvh_pos=None, lvh_neg=None):
    """Bar chart: mean |saliency| per lead. Optioneel gegroepeerd LVH+ vs LVH-."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LEAD_ORDER))
    if lvh_pos is not None and lvh_neg is not None:
        w = 0.4
        ax.bar(x - w/2, lvh_pos, w, color="crimson",   label="LVH+")
        ax.bar(x + w/2, lvh_neg, w, color="steelblue", label="LVH-")
        ax.legend(loc="upper right")
    else:
        ax.bar(x, per_lead_mean, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_ORDER)
    ax.set_ylabel("Mean |saliency|")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_mean_saliency_overlay(mean_ecg, mean_grad, title, save_path):
    """Geaggregeerde ECG-vorm + geaggregeerde saliency (rood) per lead."""
    blurred = saliency_blurred_and_scaled(mean_grad, BLUR_SIGMA)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True)
    t = np.arange(ECG_SAMPLES) / SAMPLE_RATE * 1000.0
    for k, lead in enumerate(LEAD_ORDER):
        ax = axes[k // 4, k % 4]
        ax.plot(t, mean_ecg[:, k], color="black", lw=0.7)
        ax.set_ylabel("Voltage (norm.)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(t, blurred[:, k], color="crimson", lw=0.8, alpha=0.85)
        ax2.set_ylabel("Saliency", color="crimson", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=7)
        ax.set_title(f"Lead {lead}", fontsize=10)
        if k // 4 == 2:
            ax.set_xlabel("Time (ms)", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

# --- MAIN -----------------------------------------------------------------
print("=" * 70)
print("LVM-AI Saliency Mapping - Externe validatie")
print("=" * 70)

import tensorflow as tf
print(f"\nTensorFlow {tf.__version__}")
print(f"Model laden: {MODEL_FILE}")
model = tf.keras.models.load_model(MODEL_FILE, compile=False)
n_inputs  = len(model.inputs)
n_outputs = len(model.outputs) if isinstance(model.outputs, list) else 1
print(f"  Inputs:  {n_inputs}  | Outputs: {n_outputs}")
for i, o in enumerate(model.outputs if isinstance(model.outputs, list) else [model.outputs]):
    print(f"  Output {i}: shape={o.shape}")

print(f"\nLabels laden: {LABELS_CSV}")
labels_df = pd.read_csv(LABELS_CSV)
ecg_index = {Path(p).stem: str(p) for p in Path(ECG_DIR).glob("*.xml")}
print(f"  ECG-bestanden gevonden: {len(ecg_index)}")
print(f"  Labels: {len(labels_df)}")

# Accumulatoren per groep (LVH+/LVH-) en per head (regressie/classificatie)
# Voor domain-shift evaluatie
acc = {
    "reg": {1: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0},
            0: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0}},
    "cls": {1: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0},
            0: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0}},
}
saved_indiv = {"reg": {1: 0, 0: 0}, "cls": {1: 0, 0: 0}}

has_classification = n_outputs >= 2
print(f"\nHas classification head: {has_classification}")

n_proc, n_err = 0, 0
print("\nSaliency-berekening per ECG...")
for i, row in labels_df.iterrows():
    sid = str(row["sample_id"])
    if sid not in ecg_index:
        n_err += 1; continue
    try:
        ecg     = load_xml(ecg_index[sid])
        norm    = (ecg / ECG_NORM)[np.newaxis, ...].astype(np.float32)
        age_n   = np.array([[(float(row["age"]) - AGE_MEAN) / AGE_STD]], dtype=np.float32)
        bmi_n   = np.array([[(float(row["bmi"]) - BMI_MEAN) / BMI_STD]], dtype=np.float32)
        sex_val = int(row["sex"])
        sex     = np.array([[1 - sex_val, sex_val]], dtype=np.float32)
        lvh_lbl = int(row.get("LVH", 0))

        # --- REGRESSIE-HEAD ---
        grad_reg = compute_saliency(model, norm, age_n, sex, bmi_n,
                                    head_index=0, output_index=0)
        acc["reg"][lvh_lbl]["ecg"]  += norm[0]
        acc["reg"][lvh_lbl]["grad"] += grad_reg
        acc["reg"][lvh_lbl]["n"]   += 1
        if saved_indiv["reg"][lvh_lbl] < N_INDIV_PLOTS:
            tag = "LVHpos" if lvh_lbl == 1 else "LVHneg"
            plot_saliency_per_lead(
                norm[0], grad_reg,
                f"Saliency (Regressie LVM) - {sid} - {tag}",
                os.path.join(OUTPUT_DIR, f"saliency_reg_{tag}_{sid}.png"),
            )
            saved_indiv["reg"][lvh_lbl] += 1

        # --- CLASSIFICATIE-HEAD ---
        if has_classification:
            grad_cls = compute_saliency(model, norm, age_n, sex, bmi_n,
                                        head_index=1, output_index=1)  # P(LVH=1)
            acc["cls"][lvh_lbl]["ecg"]  += norm[0]
            acc["cls"][lvh_lbl]["grad"] += grad_cls
            acc["cls"][lvh_lbl]["n"]   += 1
            if saved_indiv["cls"][lvh_lbl] < N_INDIV_PLOTS:
                tag = "LVHpos" if lvh_lbl == 1 else "LVHneg"
                plot_saliency_per_lead(
                    norm[0], grad_cls,
                    f"Saliency (Classificatie P(LVH)) - {sid} - {tag}",
                    os.path.join(OUTPUT_DIR, f"saliency_cls_{tag}_{sid}.png"),
                )
                saved_indiv["cls"][lvh_lbl] += 1

        n_proc += 1
        if n_proc % 25 == 0:
            print(f"  Verwerkt: {n_proc}")
    except Exception as e:
        n_err += 1
        if n_err <= 5:
            print(f"  Fout bij {sid}: {e}")

print(f"\nKlaar. Verwerkt: {n_proc}, fouten/mismatches: {n_err}")

# --- AGGREGATIES PER GROEP -------------------------------------------------
def finalize_group(grp):
    if grp["n"] == 0: return None, None
    return grp["ecg"] / grp["n"], grp["grad"] / grp["n"]

per_lead_importance = {}
for head_key, head_label in [("reg", "Regressie (LVM)"),
                             ("cls", "Classificatie (P(LVH))")]:
    if head_key == "cls" and not has_classification: continue
    print(f"\n--- {head_label} ---")
    mean_ecg_pos, mean_grad_pos = finalize_group(acc[head_key][1])
    mean_ecg_neg, mean_grad_neg = finalize_group(acc[head_key][0])

    if mean_grad_pos is not None:
        print(f"  LVH+: n={acc[head_key][1]['n']}")
        plot_mean_saliency_overlay(
            mean_ecg_pos, mean_grad_pos,
            f"Mean saliency LVH+ (n={acc[head_key][1]['n']}) - {head_label}",
            os.path.join(OUTPUT_DIR, f"mean_saliency_{head_key}_LVHpos.png"),
        )
    if mean_grad_neg is not None:
        print(f"  LVH-: n={acc[head_key][0]['n']}")
        plot_mean_saliency_overlay(
            mean_ecg_neg, mean_grad_neg,
            f"Mean saliency LVH- (n={acc[head_key][0]['n']}) - {head_label}",
            os.path.join(OUTPUT_DIR, f"mean_saliency_{head_key}_LVHneg.png"),
        )

    # Per-lead importance (mean abs saliency over tijd)
    imp_pos = np.abs(mean_grad_pos).mean(axis=0) if mean_grad_pos is not None else None
    imp_neg = np.abs(mean_grad_neg).mean(axis=0) if mean_grad_neg is not None else None
    per_lead_importance[head_key] = {"pos": imp_pos, "neg": imp_neg}

    if imp_pos is not None and imp_neg is not None:
        plot_per_lead_importance(
            None,
            f"Per-lead importance - {head_label}",
            os.path.join(OUTPUT_DIR, f"per_lead_importance_{head_key}.png"),
            lvh_pos=imp_pos, lvh_neg=imp_neg,
        )
        print("  Per-lead mean |saliency|:")
        print(f"    {'Lead':<5}  {'LVH+':>7}  {'LVH-':>7}  {'Ratio':>7}")
        for k, lead in enumerate(LEAD_ORDER):
            ratio = imp_pos[k] / (imp_neg[k] + 1e-12)
            print(f"    {lead:<5}  {imp_pos[k]:7.4f}  {imp_neg[k]:7.4f}  {ratio:7.3f}")

# --- TABEL EXPORT (per-lead importance, voor supplement) ------------------
rows = []
for head_key in per_lead_importance:
    imp_pos = per_lead_importance[head_key]["pos"]
    imp_neg = per_lead_importance[head_key]["neg"]
    for k, lead in enumerate(LEAD_ORDER):
        rows.append({
            "head": "regression" if head_key == "reg" else "classification",
            "lead": lead,
            "mean_abs_saliency_LVHpos": imp_pos[k] if imp_pos is not None else np.nan,
            "mean_abs_saliency_LVHneg": imp_neg[k] if imp_neg is not None else np.nan,
        })
imp_df = pd.DataFrame(rows)
imp_df.to_csv(os.path.join(OUTPUT_DIR, "per_lead_saliency_importance.csv"), index=False)
print(f"\nTabel weggeschreven: {os.path.join(OUTPUT_DIR, 'per_lead_saliency_importance.csv')}")
print(f"Output-map: {OUTPUT_DIR}/")
print("\nKlaar.")
