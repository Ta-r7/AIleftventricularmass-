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
    age_tf = tf.convert_to_tensor(age_in)
    sex_tf = tf.convert_to_tensor(sex_in)
    bmi_tf = tf.convert_to_tensor(bmi_in)
    with tf.GradientTape() as tape:
        tape.watch(ecg_tf)
        outputs = model([ecg_tf, age_tf, sex_tf, bmi_tf], training=False)
        if isinstance(outputs, (list, tuple)):
            target = outputs[head_index][:, output_index]
        else:
            target = outputs[:, output_index]
    grads = tape.gradient(target, ecg_tf).numpy()
    grads /= (np.sqrt(np.mean(grads**2)) + RMS_EPS)
    return grads[0]  # (5000, 12)

# --- POST-PROCESSING (mirrors ml4h/plots.py:_saliency_blurred_and_scaled) ---
def saliency_blurred_and_scaled(gradients, blur_radius=BLUR_SIGMA, max_value=1.0):
    # BELANGRIJK: alleen blur langs tijd-as (axis 0), niet langs lead-as (axis 1).
    # Zonder (sigma, 0) tuple zou scipy de blur ook over leads heen toepassen,
    # waardoor adjacent leads (bv. I, II, III) bijna identieke patronen krijgen.
    blurred = gaussian_filter(gradients, sigma=(blur_radius, 0))
    mn, mx = blurred.min(), blurred.max()
    if mx - mn > 1e-12:
        blurred = (blurred - mn) / (mx - mn) * max_value
    blurred -= blurred.mean()
    return blurred

# --- R-PIEK DETECTIE + UITLIJNING (voor mean plot) ------------------------
# Zonder uitlijning kanselleren QRS-complexen elkaar bij averaging omdat
# elke patient zijn QRS op een andere tijdspositie heeft. Door alle ECGs
# zo te verschuiven dat de R-piek (lead II) op target_sample valt, krijgen
# we een herkenbare gemiddelde QRS-morfologie en zinnige mean saliency.
R_PEAK_LEAD_IDX  = 1     # lead II - meest betrouwbare R-piek
R_PEAK_SEARCH    = 1500  # zoek in eerste 3 seconden
R_PEAK_TARGET    = 125   # plaats R-piek op sample 125 = ms 250 (midden van 500ms display)

def detect_r_peak(ecg_norm):
    """Sterkste positieve R-piek in lead II (eerste ~3s). Retourneert sample-index of None."""
    from scipy.signal import find_peaks
    signal = ecg_norm[:R_PEAK_SEARCH, R_PEAK_LEAD_IDX].astype(np.float64)
    s_std = float(np.std(signal))
    if s_std < 1e-9:
        return None
    peaks, props = find_peaks(signal, distance=200, prominence=s_std * 1.5)
    if len(peaks) == 0:
        # Lead II R-piek is meestal positief; fallback op absoluut signaal
        peaks, props = find_peaks(np.abs(signal), distance=200, prominence=s_std * 1.5)
        if len(peaks) == 0:
            return None
    if "prominences" in props and len(props["prominences"]) > 0:
        return int(peaks[int(np.argmax(props["prominences"]))])
    return int(peaks[0])

def accumulate_aligned(acc_entry, ecg_norm, grad, r_idx, target_sample=R_PEAK_TARGET):
    """Voeg aligned ECG + gradient toe aan accumulator.
    Verschuift signaal zodat R-piek op target_sample komt; zet wrap-zone op 0.
    Retourneert True bij succes, False als r_idx None is."""
    if r_idx is None:
        return False
    shift = target_sample - r_idx
    ecg_shifted = np.roll(ecg_norm, shift, axis=0)
    grad_shifted = np.roll(grad, shift, axis=0)
    if shift > 0:
        ecg_shifted[:shift] = 0
        grad_shifted[:shift] = 0
    elif shift < 0:
        ecg_shifted[shift:] = 0
        grad_shifted[shift:] = 0
    acc_entry["ecg"]  += ecg_shifted
    acc_entry["grad"] += grad_shifted
    acc_entry["n"]   += 1
    return True

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

# --- KHURSHID-STIJL PLOT (replicate Figure 4 origineel artikel) -----------
# Klinische 3x4 ECG-layout: rijen = limb leads + precordiaal, kolommen = paren
KHURSHID_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

def plot_saliency_khurshid_style(ecg_norm, gradients, title, save_path,
                                  display_window_ms=500,
                                  y_min=-4.0, y_max=4.0, y_ticks=(-2, 0, 2)):
    """
    Replicate Khurshid et al. Figure 4-stijl:
      * Rode ECG-lijn op klinische schaal (mV)
      * Blauwe kleur-schaduw (heatmap) als achtergrond = |saliency|
      * R/S amplitude per lead (uV) linksboven
      * Klinische 3x4 ECG-layout (I/II/III | aVR/aVL/aVF | V1-V3 | V4-V6)
      * Eerste 500 ms (een hartslag) zoals in het origineel
      * Y-as vast -4 tot +4 mV, ticks op -2, 0, 2 (conform Khurshid Figure 4)
      * Per-panel colorbar (High/Low) voor Salience-intensiteit
    ecg_norm: shape (5000, 12), in genormaliseerde eenheden (na /2000)
    gradients: shape (5000, 12), ruwe RMS-genormaliseerde gradient
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ecg_uv = ecg_norm * ECG_NORM            # terug naar microvolt
    ecg_mv = ecg_uv / 1000.0                # naar mV voor weergave
    # Blur alleen langs tijd-as om cross-lead smearing te voorkomen
    sal_abs = np.abs(gaussian_filter(gradients, sigma=(BLUR_SIGMA, 0)))

    n_show = int(display_window_ms / 1000.0 * SAMPLE_RATE)  # 500 ms = 250 samples
    n_show = min(n_show, ECG_SAMPLES)
    t = np.arange(n_show) / SAMPLE_RATE * 1000.0  # ms

    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            lead = KHURSHID_LAYOUT[r][c]
            k = LEAD_ORDER.index(lead)
            sal_lead = sal_abs[:n_show, k]
            sal_norm = sal_lead / sal_lead.max() if sal_lead.max() > 0 else sal_lead
            ecg_lead_mv = ecg_mv[:n_show, k]

            # Blauwe heatmap achtergrond (saliency) - vult de hele y-range
            im = ax.imshow(
                sal_norm[np.newaxis, :],
                extent=[t[0], t[-1], y_min, y_max],
                cmap="Blues", aspect="auto",
                alpha=0.75, vmin=0, vmax=1, zorder=1,
            )
            # Rode ECG-lijn erbovenop
            ax.plot(t, ecg_lead_mv, color="red", lw=0.9, zorder=2)

            # R/S amplitudes in microvolt
            R = int(round(float(ecg_uv[:n_show, k].max())))
            S = int(round(abs(float(ecg_uv[:n_show, k].min()))))
            ax.text(0.02, 0.97, f"R:{R} S:{S}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="none", alpha=0.8),
                    zorder=3)
            ax.set_title(f"strip_{lead}", fontsize=10)
            ax.set_ylabel("mV", fontsize=8)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(list(y_ticks))
            ax.tick_params(axis="both", labelsize=7)
            if r == 2:
                ax.set_xlabel("milliseconds", fontsize=8)

            # Per-panel Salience colorbar (High/Low) conform Khurshid Figure 4
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.04)
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 1])
            cbar.ax.set_yticklabels(["Low", "High"], fontsize=6)
            cbar.set_label("Salience", fontsize=7, rotation=270, labelpad=8)
            cbar.ax.tick_params(labelsize=6, length=2)
            cbar.outline.set_linewidth(0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_mean_saliency_khurshid_style(mean_ecg_norm, mean_grad, title, save_path,
                                       display_window_ms=500,
                                       y_min=-4.0, y_max=4.0, y_ticks=(-2, 0, 2)):
    """
    Khurshid-stijl voor GEAGGREGEERDE saliency over een groep (bv. alle LVH+).
    Zelfde visualisatie als plot_saliency_khurshid_style maar met gemiddelden.
    """
    plot_saliency_khurshid_style(
        mean_ecg_norm, mean_grad, title, save_path,
        display_window_ms=display_window_ms,
        y_min=y_min, y_max=y_max, y_ticks=y_ticks,
    )

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
labels_df = pd.read_csv(LABELS_CSV, sep=None, engine="python", encoding="utf-8-sig")  # auto-detect delimiter + strip BOM
ecg_index = {Path(p).stem: str(p) for p in Path(ECG_DIR).glob("*.xml")}
print(f"  ECG-bestanden gevonden: {len(ecg_index)}")
print(f"  Labels: {len(labels_df)}")
print(f"  Kolommen in CSV: {list(labels_df.columns)}")

# Helpers voor Europese decimaalnotatie (komma -> punt)
def to_float(v):
    if isinstance(v, str):
        v = v.replace(",", ".")
    return float(v)

# Auto-detect kolomnamen (case-insensitive)
def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

ID_COL  = find_col(labels_df, ["sample_id", "id", "patient_id", "mrn", "study_id", "filename", "subject_id"])
AGE_COL = find_col(labels_df, ["age", "leeftijd", "age_years"])
BMI_COL = find_col(labels_df, ["bmi", "body_mass_index"])
SEX_COL = find_col(labels_df, ["sex", "geslacht", "gender", "male"])
LVH_COL = find_col(labels_df, ["LVH", "lvh", "lvh_label", "lvh_status"])

if ID_COL is None:
    raise KeyError(f"Geen ID-kolom gevonden in CSV. Beschikbaar: {list(labels_df.columns)}. "
                   f"Voeg een kolom toe met naam 'sample_id' (of pas find_col aan).")
print(f"  Gebruik kolommen -> id:{ID_COL}, age:{AGE_COL}, bmi:{BMI_COL}, sex:{SEX_COL}, LVH:{LVH_COL}")

# Accumulatoren per groep (LVH+/LVH-) en per head (regressie/classificatie)
# Voor domain-shift evaluatie
acc = {
    "reg": {1: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0},
            0: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0}},
    "cls": {1: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0},
            0: {"ecg": np.zeros((ECG_SAMPLES, 12)), "grad": np.zeros((ECG_SAMPLES, 12)), "n": 0}},
}
saved_indiv = {"reg": {1: 0, 0: 0}, "cls": {1: 0, 0: 0}}

# Per-patient dominante lead (zoals Khurshid et al.: "V5 in 97.4%, V4 in 2.4%, V1 in 0.3%")
dominant_leads = {"reg": [], "cls": []}     # lijsten van (sample_id, dominant_lead, lvh_lbl)
per_patient_imp = {"reg": [], "cls": []}    # rij per patient met mean |saliency| per lead

has_classification = n_outputs >= 2
print(f"\nHas classification head: {has_classification}")

n_proc, n_err, n_aligned, n_align_fail = 0, 0, 0, 0
print("\nSaliency-berekening per ECG...")
for i, row in labels_df.iterrows():
    sid = str(row[ID_COL])
    if sid not in ecg_index:
        n_err += 1; continue
    try:
        ecg     = load_xml(ecg_index[sid])
        norm    = (ecg / ECG_NORM)[np.newaxis, ...].astype(np.float32)
        age_n   = np.array([[(to_float(row[AGE_COL]) - AGE_MEAN) / AGE_STD]], dtype=np.float32)
        bmi_n   = np.array([[(to_float(row[BMI_COL]) - BMI_MEAN) / BMI_STD]], dtype=np.float32)
        sex_val = int(to_float(row[SEX_COL]))
        sex     = np.array([[1 - sex_val, sex_val]], dtype=np.float32)
        lvh_lbl = int(to_float(row[LVH_COL])) if LVH_COL is not None else 0

        # R-piek detectie (eenmaal per patient, gebruikt voor beide heads)
        r_idx = detect_r_peak(norm[0])
        if r_idx is None:
            n_align_fail += 1
        else:
            n_aligned += 1

        # --- REGRESSIE-HEAD ---
        grad_reg = compute_saliency(model, norm, age_n, sex, bmi_n,
                                    head_index=0, output_index=0)
        # Accumulator update met aligned data (skip als R-piek detectie mislukt)
        accumulate_aligned(acc["reg"][lvh_lbl], norm[0], grad_reg, r_idx)
        # Per-patient lead-importance + dominante lead (uit niet-uitgelijnde data)
        imp_per_lead_reg = np.abs(grad_reg).mean(axis=0)
        dom_lead_reg = LEAD_ORDER[int(np.argmax(imp_per_lead_reg))]
        dominant_leads["reg"].append((sid, dom_lead_reg, lvh_lbl))
        per_patient_imp["reg"].append([sid, lvh_lbl] + list(imp_per_lead_reg))
        if saved_indiv["reg"][lvh_lbl] < N_INDIV_PLOTS:
            tag = "LVHpos" if lvh_lbl == 1 else "LVHneg"
            plot_saliency_per_lead(
                norm[0], grad_reg,
                f"Saliency (Regressie LVM) - {sid} - {tag}",
                os.path.join(OUTPUT_DIR, f"saliency_reg_{tag}_{sid}.png"),
            )
            # Khurshid-stijl Figure 4 replica (individuele plots NIET aligned)
            plot_saliency_khurshid_style(
                norm[0], grad_reg,
                f"LVM-AI saliency map - {sid} - {tag}",
                os.path.join(OUTPUT_DIR, f"saliency_reg_KHURSHID_{tag}_{sid}.png"),
            )
            saved_indiv["reg"][lvh_lbl] += 1

        # --- CLASSIFICATIE-HEAD ---
        if has_classification:
            grad_cls = compute_saliency(model, norm, age_n, sex, bmi_n,
                                        head_index=1, output_index=1)  # P(LVH=1)
            accumulate_aligned(acc["cls"][lvh_lbl], norm[0], grad_cls, r_idx)
            imp_per_lead_cls = np.abs(grad_cls).mean(axis=0)
            dom_lead_cls = LEAD_ORDER[int(np.argmax(imp_per_lead_cls))]
            dominant_leads["cls"].append((sid, dom_lead_cls, lvh_lbl))
            per_patient_imp["cls"].append([sid, lvh_lbl] + list(imp_per_lead_cls))
            if saved_indiv["cls"][lvh_lbl] < N_INDIV_PLOTS:
                tag = "LVHpos" if lvh_lbl == 1 else "LVHneg"
                plot_saliency_per_lead(
                    norm[0], grad_cls,
                    f"Saliency (Classificatie P(LVH)) - {sid} - {tag}",
                    os.path.join(OUTPUT_DIR, f"saliency_cls_{tag}_{sid}.png"),
                )
                plot_saliency_khurshid_style(
                    norm[0], grad_cls,
                    f"LVM-AI saliency map (P(LVH)) - {sid} - {tag}",
                    os.path.join(OUTPUT_DIR, f"saliency_cls_KHURSHID_{tag}_{sid}.png"),
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
print(f"R-piek detectie geslaagd: {n_aligned} / {n_proc} (mislukt: {n_align_fail})")

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
        # Khurshid-stijl mean plot (Figure 4-stijl over de groep)
        plot_mean_saliency_khurshid_style(
            mean_ecg_pos, mean_grad_pos,
            f"Mean LVM-AI saliency map - LVH+ (n={acc[head_key][1]['n']}) - {head_label}",
            os.path.join(OUTPUT_DIR, f"mean_saliency_KHURSHID_{head_key}_LVHpos.png"),
        )
    if mean_grad_neg is not None:
        print(f"  LVH-: n={acc[head_key][0]['n']}")
        plot_mean_saliency_overlay(
            mean_ecg_neg, mean_grad_neg,
            f"Mean saliency LVH- (n={acc[head_key][0]['n']}) - {head_label}",
            os.path.join(OUTPUT_DIR, f"mean_saliency_{head_key}_LVHneg.png"),
        )
        plot_mean_saliency_khurshid_style(
            mean_ecg_neg, mean_grad_neg,
            f"Mean LVM-AI saliency map - LVH- (n={acc[head_key][0]['n']}) - {head_label}",
            os.path.join(OUTPUT_DIR, f"mean_saliency_KHURSHID_{head_key}_LVHneg.png"),
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

# --- DOMINANTE LEAD PER PATIENT (replicate Khurshid: "V5 in 97.4%, V4 in 2.4%, V1 in 0.3%") ---
print("\n" + "=" * 70)
print("DOMINANTE LEAD PER PATIENT (frequentie-analyse)")
print("=" * 70)

for head_key, head_label in [("reg", "Regressie (LVM)"),
                             ("cls", "Classificatie (P(LVH))")]:
    if head_key == "cls" and not has_classification: continue
    dl = dominant_leads[head_key]
    if not dl: continue
    print(f"\n--- {head_label} (n={len(dl)} patienten) ---")

    # Frequentie-tabel: hele cohort
    dom_df = pd.DataFrame(dl, columns=["sample_id", "dominant_lead", "LVH"])
    counts = dom_df["dominant_lead"].value_counts()
    pct    = (counts / len(dom_df) * 100).round(2)
    print("\n  Hele cohort:")
    for lead, n in counts.items():
        print(f"    {lead:<5}  n={n:>4}  ({pct[lead]:.1f}%)")

    # Frequentie-tabel: LVH+ / LVH-
    for grp_lbl, grp_name in [(1, "LVH+"), (0, "LVH-")]:
        sub = dom_df[dom_df["LVH"] == grp_lbl]
        if len(sub) == 0: continue
        counts_s = sub["dominant_lead"].value_counts()
        pct_s    = (counts_s / len(sub) * 100).round(2)
        print(f"\n  {grp_name} (n={len(sub)}):")
        for lead, n in counts_s.items():
            print(f"    {lead:<5}  n={n:>4}  ({pct_s[lead]:.1f}%)")

    dom_df.to_csv(os.path.join(OUTPUT_DIR, f"dominant_lead_per_patient_{head_key}.csv"), index=False)

    # Bar chart: frequentie dominante lead in cohort
    fig, ax = plt.subplots(figsize=(10, 5))
    cohort_counts = dom_df["dominant_lead"].value_counts().reindex(LEAD_ORDER, fill_value=0)
    cohort_pct    = cohort_counts / len(dom_df) * 100
    ax.bar(np.arange(len(LEAD_ORDER)), cohort_pct.values, color="dimgray")
    ax.set_xticks(np.arange(len(LEAD_ORDER)))
    ax.set_xticklabels(LEAD_ORDER)
    ax.set_ylabel("% patienten waar lead dominant is")
    ax.set_title(f"Dominante lead per patient - {head_label} (n={len(dom_df)})")
    ax.grid(axis="y", alpha=0.3)
    for k, v in enumerate(cohort_pct.values):
        if v > 0.5:
            ax.text(k, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"dominant_lead_frequency_{head_key}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Volledige per-patient lead-importance matrix
    cols = ["sample_id", "LVH"] + LEAD_ORDER
    pat_imp_df = pd.DataFrame(per_patient_imp[head_key], columns=cols)
    pat_imp_df.to_csv(os.path.join(OUTPUT_DIR, f"per_patient_lead_importance_{head_key}.csv"), index=False)

print(f"\nOutput-map: {OUTPUT_DIR}/")
print("\nKlaar.")
