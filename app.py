"""
ECG Heartbeat Classification — Streamlit App
============================================
Complete pipeline:
  1. Data Generation  — synthetic MIT-BIH style ECG beats (5 classes)
  2. Feature Extraction — temporal feature sequences for HMM
  3. HMM Training     — one Gaussian HMM per class (Baum-Welch)
  4. Classification   — log-likelihood scoring
  5. Evaluation       — accuracy, confusion matrix, per-class report
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import streamlit as st

from ecg_data import (
    generate_dataset, generate_beat, get_class_info,
    normalize_beat, BEAT_LENGTH, CLASS_NAMES, CLASS_COLORS
)
from features import extract_sequence, extract_dataset, feature_info, T_STEPS
from hmm_model import HMMClassifier


# =========================================================================== #
#  Page configuration
# =========================================================================== #
st.set_page_config(
    page_title="ECG Heartbeat Classifier — HMM",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================== #
#  CSS styling
# =========================================================================== #
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#e74c3c; }
    .section-hdr { font-size:1.5rem; font-weight:700; margin-top:0.5rem; }
    .metric-box  {
        background: #1e1e2e; border-radius:10px; padding:1rem;
        text-align:center; margin:4px;
    }
    .metric-val  { font-size:2rem; font-weight:700; color:#e74c3c; }
    .metric-lbl  { font-size:0.8rem; color:#aaa; }
    .class-badge {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:0.85rem; font-weight:600; margin:2px;
    }
    .status-ok   { background:#2ecc71; color:#000; }
    .status-warn { background:#f39c12; color:#000; }
    .stButton>button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; border:none; border-radius:8px;
        font-weight:600; padding:0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================== #
#  Sidebar — Navigation & Settings
# =========================================================================== #
with st.sidebar:
    st.markdown("## 🫀 ECG-HMM Classifier")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Data Overview", "⚙️ Preprocessing", "🏋️ Train HMM",
         "🔬 Classify Beat", "📈 Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Dataset Settings")
    n_per_class = st.slider("Samples per class", 100, 500, 200, 50)
    test_size   = st.slider("Test split (%)", 10, 40, 20, 5) / 100
    seed        = st.number_input("Random seed", value=42, step=1)

# قيم ثابتة بدل الـ HMM sliders
n_states = 4
n_iter   = 60
t_steps  = T_STEPS

# =========================================================================== #
#  Session-state helpers
# =========================================================================== #
def get_state(key, default=None):
    return st.session_state.get(key, default)


def set_state(key, val):
    st.session_state[key] = val


# =========================================================================== #
#  Shared: generate / cache dataset
# =========================================================================== #
@st.cache_data(show_spinner=False)
def cached_dataset(n_per_class, seed):
    beats, labels = generate_dataset(n_per_class=n_per_class, seed=seed)
    return beats, labels


def get_dataset():
    beats, labels = cached_dataset(n_per_class, int(seed))
    return beats, labels


# =========================================================================== #
#  Colour helpers
# =========================================================================== #
def cls_color(cls):
    return CLASS_COLORS.get(cls, "#888")


# =========================================================================== #
#  PAGE 1 — Data Overview
# =========================================================================== #
if page == "📊 Data Overview":
    st.markdown('<p class="main-title">📊 ECG Data Overview</p>', unsafe_allow_html=True)
    st.markdown(
        "Synthetic heartbeats modelled on the **MIT-BIH Arrhythmia Database** "
        "(5 arrhythmia classes, 180 samples / beat at 360 Hz)."
    )

    beats, labels = get_dataset()
    info = get_class_info()

    # ----- class distribution ----- #
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Class Distribution")
        unique, counts = np.unique(labels, return_counts=True)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bars = ax.bar(unique, counts,
                      color=[cls_color(c) for c in unique],
                      edgecolor="white", linewidth=0.8)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(cnt), ha="center", va="bottom", color="white", fontsize=9)
        ax.set_ylabel("Count", color="white")
        ax.set_xlabel("Class", color="white")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Class Legend")
        for cls in info["classes"]:
            color = cls_color(cls)
            st.markdown(
                f'<div style="background:{color};color:#000;padding:6px 12px;'
                f'border-radius:8px;margin:4px;font-weight:600;">'
                f'<b>{cls}</b> — {CLASS_NAMES[cls]}</div>',
                unsafe_allow_html=True,
            )
        st.metric("Total samples", len(beats))
        st.metric("Beat length", f"{BEAT_LENGTH} pts")
        st.metric("Sampling rate", "360 Hz")

    st.markdown("---")
    st.subheader("Sample Heartbeats per Class")

    rng_demo = np.random.RandomState(7)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=False)
    fig.patch.set_facecolor("#0e1117")
    t = np.arange(BEAT_LENGTH) / 360 * 1000   # ms

    for ax, cls in zip(axes, info["classes"]):
        sample = generate_beat(cls, rng_demo)
        ax.plot(t, sample, color=cls_color(cls), linewidth=1.5)
        ax.set_title(f"{cls}\n{CLASS_NAMES[cls]}", color="white", fontsize=8)
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#333")
        if cls == "N":
            ax.set_ylabel("Amplitude (mV)", color="white", fontsize=8)
        ax.set_xlabel("Time (ms)", color="white", fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ----- overlay view ----- #
    st.markdown("---")
    st.subheader("Class Overlay (mean ± std)")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e2e")

    for cls in info["classes"]:
        cls_beats = [beats[i] for i in range(len(beats)) if labels[i] == cls]
        mat = np.stack([normalize_beat(b) for b in cls_beats[:80]])
        mean, std = mat.mean(0), mat.std(0)
        c = cls_color(cls)
        ax.plot(t, mean, color=c, linewidth=2, label=f"{cls} — {CLASS_NAMES[cls]}")
        ax.fill_between(t, mean - std, mean + std, color=c, alpha=0.15)

    ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
    ax.set_xlabel("Time (ms)", color="white")
    ax.set_ylabel("Norm. Amplitude", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================================== #
#  PAGE 2 — Preprocessing
# =========================================================================== #
elif page == "⚙️ Preprocessing":
    st.markdown('<p class="main-title">⚙️ Preprocessing & Feature Extraction</p>',
                unsafe_allow_html=True)

    beats, labels = get_dataset()
    fi = feature_info()

    st.info(
        f"Each raw beat ({BEAT_LENGTH} samples) is **z-score normalised** then "
        f"converted to a **({t_steps} × {fi['n_features']})** observation matrix "
        f"by computing 6 features per temporal block."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Input length",  f"{BEAT_LENGTH} samples")
    col2.metric("Temporal steps", t_steps)
    col3.metric("Features / step", fi['n_features'])

    st.markdown("### Feature Descriptions")
    feat_desc = {
        "Amplitude"         : "Z-score normalised signal value — captures morphology",
        "1st Derivative"    : "Rate of change — detects onset/offset of waves",
        "2nd Derivative"    : "Curvature — highlights peaks and inflection points",
        "Envelope (|amp|)"  : "Absolute amplitude — energy envelope",
        "Rolling Energy"    : "Local mean squared amplitude — short-term power",
        "Cumulative Energy" : "Running mean of squared amplitude — trend over beat",
    }
    for fname, desc in feat_desc.items():
        st.markdown(f"- **{fname}**: {desc}")

    st.markdown("---")
    st.subheader("Visualise a Single Beat Pipeline")

    demo_cls = st.selectbox("Select beat class", list(CLASS_NAMES.keys()),
                             format_func=lambda x: f"{x} — {CLASS_NAMES[x]}")
    rng_d = np.random.RandomState(int(seed) + 99)
    raw   = generate_beat(demo_cls, rng_d)
    norm  = normalize_beat(raw)
    obs   = extract_sequence(raw, t_steps=t_steps)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    t_full = np.arange(BEAT_LENGTH) / 360 * 1000
    t_obs  = np.linspace(0, t_full[-1], t_steps)

    # Raw
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_full, raw, color="#aaa", linewidth=1.2, label="Raw")
    ax0.plot(t_full, norm, color=cls_color(demo_cls), linewidth=1.5,
             label="Normalised")
    ax0.set_facecolor("#1e1e2e"); ax0.tick_params(colors="white")
    ax0.spines[:].set_color("#333")
    ax0.set_title("Raw → Normalised", color="white")
    ax0.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
    ax0.set_xlabel("Time (ms)", color="white")

    feat_names = fi["feature_names"]
    feat_colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    positions = [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    for idx, (row, col_) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col_])
        ax.plot(t_obs, obs[:, idx], color=feat_colors[idx], linewidth=1.5,
                marker="o", markersize=3)
        ax.set_title(feat_names[idx], color="white", fontsize=8)
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white", labelsize=6)
        ax.spines[:].set_color("#333")
        ax.set_xlabel("ms", color="white", fontsize=7)

    st.pyplot(fig)
    plt.close(fig)


# =========================================================================== #
#  PAGE 3 — Train HMM
# =========================================================================== #
elif page == "🏋️ Train HMM":
    st.markdown('<p class="main-title">🏋️ Train HMM Classifiers</p>',
                unsafe_allow_html=True)

    beats, labels = get_dataset()

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        beats, labels, test_size=test_size,
        random_state=int(seed), stratify=labels,
    )

    st.info(
        f"**Training set**: {len(X_train_raw)} beats  |  "
        f"**Test set**: {len(X_test_raw)} beats"
    )

    # Store split in session
    set_state("X_test_raw", X_test_raw)
    set_state("y_test",      y_test)

    if st.button("🚀 Start Training"):
        with st.spinner("Extracting features …"):
            X_train = extract_dataset(X_train_raw, t_steps=t_steps)
            X_test  = extract_dataset(X_test_raw,  t_steps=t_steps)

        progress_bar = st.progress(0)
        status_txt   = st.empty()

        clf       = HMMClassifier(n_states=n_states, n_iter=n_iter)
        ll_curves = {}
        classes   = sorted(set(y_train))

        t_start = time.time()
        for idx, cls in enumerate(classes):
            status_txt.markdown(
                f"⏳ Training **{cls}** ({CLASS_NAMES[cls]}) — "
                f"{idx+1}/{len(classes)} …"
            )
            seqs = [X_train[i] for i in range(len(X_train)) if y_train[i] == cls]

            from hmm_model import GaussianHMM
            hmm = GaussianHMM(n_states=n_states, n_iter=n_iter,
                              random_state=hash(cls) % (2**31))
            hmm.fit(seqs)
            clf.models_[cls]  = hmm
            clf.classes_      = classes
            clf.is_fitted     = True
            ll_curves[cls]    = hmm.ll_history_
            progress_bar.progress((idx + 1) / len(classes))

        t_elapsed = time.time() - t_start
        status_txt.success(f"✅ Training complete in {t_elapsed:.1f} s")

        # Predict on test set
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        set_state("clf",       clf)
        set_state("y_pred",    y_pred)
        set_state("acc",       acc)
        set_state("ll_curves", ll_curves)
        set_state("X_test",    X_test)
        set_state("t_steps",   t_steps)

        # Metrics summary
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{acc*100:.2f}%")
        col2.metric("Classes trained", len(classes))
        col3.metric("Training time", f"{t_elapsed:.1f} s")

        # LL convergence curves
        st.subheader("EM Convergence (log-likelihood per class)")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e1e2e")
        for cls, curve in ll_curves.items():
            ax.plot(curve, label=f"{cls} — {CLASS_NAMES[cls]}",
                    color=cls_color(cls), linewidth=1.8, marker="o",
                    markersize=3)
        ax.set_xlabel("EM Iteration", color="white")
        ax.set_ylabel("Total Log-Likelihood", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif get_state("clf") is not None:
        acc = get_state("acc")
        st.success(
            f"✅ Model already trained — Test accuracy: **{acc*100:.2f}%**  "
            "| Navigate to **Classify Beat** or **Evaluation** for details."
        )
    else:
        st.warning("👆 Press **Start Training** to begin.")


# =========================================================================== #
#  PAGE 4 — Classify Beat
# =========================================================================== #
elif page == "🔬 Classify Beat":
    st.markdown('<p class="main-title">🔬 Beat Classification</p>',
                unsafe_allow_html=True)

    clf = get_state("clf")
    if clf is None or not clf.is_fitted:
        st.warning("⚠️ No trained model found. Please go to **🏋️ Train HMM** first.")
        st.stop()

    st.info("Upload a heartbeat signal file (CSV/TXT/NPY) containing exactly 180 values, and the model will classify it.")

    uploaded_file = st.file_uploader(
        "Upload beat file",
        type=["csv", "txt", "npy"]
    )

    raw = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".npy"):
                raw = np.load(uploaded_file, allow_pickle=False)
            else:
                try:
                    raw = np.loadtxt(uploaded_file, delimiter=",")
                except:
                    uploaded_file.seek(0)
                    raw = np.loadtxt(uploaded_file)

                if raw.ndim > 1:
                    raw = raw.flatten()

            raw = np.array(raw, dtype=np.float32)

            if len(raw) != BEAT_LENGTH:
                st.error(
                    f"❌ Input beat must contain exactly {BEAT_LENGTH} samples. "
                    f"Uploaded length = {len(raw)}"
                )
                raw = None

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
            raw = None

    if raw is not None:
        obs   = extract_sequence(raw, t_steps=get_state("t_steps") or T_STEPS)
        proba = clf.predict_proba([obs])[0]
        pred  = clf.classes_[np.argmax(proba)]

        st.success(f"✅ Predicted Class: {pred} — {CLASS_NAMES[pred]}")

        # Waveform plot
        t = np.arange(BEAT_LENGTH) / 360 * 1000
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e1e2e")
        ax.plot(t, raw, color=cls_color(pred), linewidth=1.8)
        ax.set_title(f"Uploaded beat — Predicted: {pred}", color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.set_xlabel("Time (ms)", color="white")
        ax.set_ylabel("Amplitude", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Probability chart
        fig2, ax2 = plt.subplots(figsize=(8, 2.8))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#1e1e2e")
        bars = ax2.barh(clf.classes_, proba * 100,
                        color=[cls_color(c) for c in clf.classes_],
                        edgecolor="white", linewidth=0.5)
        for bar, p in zip(bars, proba):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{p*100:.1f}%", va="center", color="white", fontsize=9)
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Probability (%)", color="white")
        ax2.set_title("Class Probabilities", color="white", fontsize=10)
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info(f"👈 Please upload a file containing exactly {BEAT_LENGTH} ECG samples.")


# =========================================================================== #
#  PAGE 5 — Evaluation
# =========================================================================== #
elif page == "📈 Evaluation":
    st.markdown('<p class="main-title">📈 Model Evaluation</p>', unsafe_allow_html=True)

    clf    = get_state("clf")
    y_pred = get_state("y_pred")
    y_test = get_state("y_test")
    acc    = get_state("acc")

    if clf is None or y_pred is None:
        st.warning("⚠️ No results found. Please run **🏋️ Train HMM** first.")
        st.stop()

    classes = clf.classes_

    # ---- Top metrics ---- #
    st.subheader("📌 Overall Metrics")
    report = classification_report(y_test, y_pred, output_dict=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",         f"{acc*100:.2f}%")
    c2.metric("Macro Precision",  f"{report['macro avg']['precision']*100:.2f}%")
    c3.metric("Macro Recall",     f"{report['macro avg']['recall']*100:.2f}%")
    c4.metric("Macro F1",         f"{report['macro avg']['f1-score']*100:.2f}%")

    # ---- Confusion matrix ---- #
    st.markdown("---")
    col_cm, col_pr = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e1e2e")
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color="white")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color="white")
        ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
        ax.set_title("Confusion Matrix", color="white")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black" if cm[i, j] > cm.max() * 0.5 else "white",
                        fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_pr:
        st.subheader("Per-Class Report")
        rows = []
        for cls in classes:
            r = report.get(cls, {})
            rows.append({
                "Class"     : cls,
                "Name"      : CLASS_NAMES[cls],
                "Precision" : f"{r.get('precision',0)*100:.1f}%",
                "Recall"    : f"{r.get('recall',0)*100:.1f}%",
                "F1-Score"  : f"{r.get('f1-score',0)*100:.1f}%",
                "Support"   : int(r.get("support", 0)),
            })
        df_rep = pd.DataFrame(rows)
        st.dataframe(df_rep, use_container_width=True, hide_index=True)

        # Per-class accuracy bar
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#1e1e2e")
        accs = [report[c]["recall"] * 100 for c in classes]
        bars = ax3.bar(classes, accs, color=[cls_color(c) for c in classes],
                       edgecolor="white", linewidth=0.5)
        ax3.axhline(y=acc * 100, color="white", linestyle="--",
                    linewidth=1, label=f"Avg {acc*100:.1f}%")
        for bar, a in zip(bars, accs):
            ax3.text(bar.get_x() + bar.get_width()/2, a + 1,
                     f"{a:.1f}%", ha="center", color="white", fontsize=8)
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Recall (%)", color="white")
        ax3.set_title("Per-Class Recall", color="white")
        ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333")
        ax3.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ---- Error analysis ---- #
    st.markdown("---")
    st.subheader("🔍 Error Analysis")

    X_test  = get_state("X_test")

    wrong_idx = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt != yp]
    st.metric("Misclassified beats", len(wrong_idx),
              delta=f"-{len(wrong_idx)} errors")

    if wrong_idx:
        X_test_raw_all = get_state("X_test_raw")
        show_n = min(6, len(wrong_idx))
        fig4, axes4 = plt.subplots(1, show_n, figsize=(14, 2.8), sharey=False)
        fig4.patch.set_facecolor("#0e1117")
        if show_n == 1:
            axes4 = [axes4]
        for ax, idx in zip(axes4, wrong_idx[:show_n]):
            raw = X_test_raw_all[idx]
            t   = np.arange(BEAT_LENGTH) / 360 * 1000
            ax.plot(t, raw, color=cls_color(y_test[idx]), linewidth=1.2)
            ax.set_title(
                f"True: {y_test[idx]}\nPred: {y_pred[idx]}",
                color="white", fontsize=7,
            )
            ax.set_facecolor("#1e1e2e")
            ax.tick_params(colors="white", labelsize=6)
            ax.spines[:].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

