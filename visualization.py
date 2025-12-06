import numpy as np
import matplotlib.pyplot as plt
import wfdb
from typing import Optional, Tuple


# Standard 12-lead names
LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def load_ecg_waveform(
    ecg_id: int,
    ptbxl_path: str,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Load a 12-lead ECG waveform from PTB-XL given an ECG ID and base path.

    Args:
        ecg_id: Integer PTB-XL record ID.
        ptbxl_path: Base PTB-XL path, e.g. "/content/data/ptb-xl".

    Returns:
        signals: np.ndarray of shape (n_samples, n_leads) or None on failure.
        fs: Sampling frequency (Hz) or None on failure.
    """
    subdir = f"{(ecg_id // 1000) * 1000:05d}"
    path = f"{ptbxl_path}/records500/{subdir}/{ecg_id:05d}_hr"

    try:
        signals, fields = wfdb.rdsamp(path)
        fs = float(fields["fs"])
        return signals, fs
    except Exception as e:
        print(f"[ERROR] Could not load ECG {ecg_id} from '{path}': {e}")
        return None, None


def visualize_12_lead(
    ecg_id: int,
    ptbxl_path: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    prediction: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize a 12-lead ECG with optional QA context.

    Args:
        ecg_id: Integer PTB-XL record ID.
        ptbxl_path: Base PTB-XL path, e.g. "/content/data/ptb-xl".
        question: Optional question string to show in the title.
        answer: Optional ground-truth answer.
        prediction: Optional model prediction.
        save_path: If provided, save the figure to this path instead of showing.
    """
    signals, fs = load_ecg_waveform(ecg_id, ptbxl_path)
    if signals is None or fs is None:
        print(f"[ERROR] Could not load ECG {ecg_id}")
        return

    n_samples = signals.shape[0]
    n_leads = min(12, signals.shape[1])
    duration = n_samples / fs
    gap_samples = int(0.2 * fs)

    # Normalize each lead independently
    signals_norm = np.array(
        [(s - np.mean(s)) / (np.std(s) + 1e-8) for s in signals.T]
    ).T

    fig, ax = plt.subplots(figsize=(22, 3.5))

    # Color groups: limb, augmented limb, precordial
    colors = ["#1976D2"] * 3 + ["#388E3C"] * 3 + ["#D32F2F"] * 6

    x_offset = 0
    for i in range(n_leads):
        t = np.arange(n_samples) / fs + x_offset / fs
        ax.plot(t, signals_norm[:, i], color=colors[i], linewidth=0.5)

        # Lead label above each segment
        ax.text(
            (x_offset + n_samples / 2) / fs,
            3.5,
            LEAD_NAMES[i],
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=colors[i],
        )

        # Vertical separator line between leads
        if i < n_leads - 1:
            ax.axvline(
                (x_offset + n_samples + gap_samples / 2) / fs,
                color="#E0E0E0",
                lw=1,
            )

        x_offset += n_samples + gap_samples

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (norm)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, x_offset / fs])
    ax.set_facecolor("#FAFAFA")

    # Build title with optional QA context
    title = f"[ECG] 12-Lead | ID: {ecg_id} | {fs:.0f} Hz | {duration:.1f}s/lead"

    if question:
        title += f"\n[Q]: {question[:70]}"

    if answer is not None and prediction is not None:
        correct = str(answer).lower().strip() == str(prediction).lower().strip()
        status = "[CORRECT]" if correct else "[INCORRECT]"
        title += f"\n[GT]: {answer} | [Pred]: {prediction} | {status}"
        ax.set_facecolor("#F1F8E9" if correct else "#FFEBEE")
    elif answer is not None:
        title += f" | [A]: {answer}"

    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
