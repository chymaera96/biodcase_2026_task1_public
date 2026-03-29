import numpy as np
from torch import nn
import torchaudio
import yaml
from types import SimpleNamespace


def load_hyperparams(config_fp: str) -> SimpleNamespace:
    """Load a flat YAML mapping into a SimpleNamespace.

    Assumes configs are always available and are not nested.
    """

    if config_fp is None:
        raise ValueError("config_fp must be provided")
    with open(str(config_fp), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/dict, got {type(data)} from {config_fp}")
    return SimpleNamespace(**data)


def require_hyperparams(hp: SimpleNamespace, keys: list[str], *, config_fp: str) -> None:
    missing = [k for k in keys if not hasattr(hp, k)]
    if missing:
        raise ValueError(f"Missing hyperparam(s) in {config_fp}: {missing}")


def make_offset_grid(max_allowed_error: float, offset_step: float) -> np.ndarray:
    """Create a symmetric offset grid in seconds.

    Args:
        max_allowed_error: maximum absolute offset (seconds).
        offset_step: grid step (seconds), must be > 0.

    Returns:
        1D numpy array of offsets in seconds, ascending.
    """
    max_allowed_error = float(max_allowed_error)
    offset_step = float(offset_step)
    if max_allowed_error < 0:
        raise ValueError("max_allowed_error must be >= 0")
    if offset_step <= 0:
        raise ValueError("offset_step must be > 0")
    if max_allowed_error == 0:
        return np.array([0.0], dtype=float)
    return np.arange(-max_allowed_error, max_allowed_error + 0.5 * offset_step, offset_step, dtype=float)


def normalize_scores_per_keypoint(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each row (keypoint) of a score matrix for stable DP smoothing.

    Uses per-row z-score normalization after subtracting the row max to keep numbers
    well-scaled even when raw scores have different magnitudes across keypoints.

    Args:
        scores: shape [T, M] array of scores; larger is better.
        eps: numerical stability constant.

    Returns:
        shape [T, M] normalized scores.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2:
        raise ValueError("scores must be a 2D array")

    # Handle rows where some/all entries are non-finite (e.g., invalid offsets -> -inf).
    s2 = s.copy()
    for i in range(s2.shape[0]):
        row = s2[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            # No information for this keypoint: make it neutral.
            row[:] = 0.0
        else:
            worst = float(np.min(row[finite]))
            row[~finite] = worst

    s2 = s2 - np.max(s2, axis=1, keepdims=True)
    mu = np.mean(s2, axis=1, keepdims=True)
    sigma = np.std(s2, axis=1, keepdims=True)
    return (s2 - mu) / (sigma + float(eps))


def viterbi_smooth_offsets(
    offsets: np.ndarray,
    scores: np.ndarray,
    smoothness_lambda: float,
) -> np.ndarray:
    """Choose a smooth(ish) offset path that maximizes total score.

    This performs Viterbi/DP over a discrete offset grid. It encourages temporal
    consistency while still allowing step changes when the evidence is strong.

    Objective:
        maximize sum_t score[t, j_t] - smoothness_lambda * |offset[j_t] - offset[j_{t-1}]|

    Args:
        offsets: shape [M] array of candidate offsets in seconds (ascending).
        scores: shape [T, M] array of per-keypoint scores (larger is better).
        smoothness_lambda: >= 0; 0 disables smoothing (pure per-keypoint argmax).

    Returns:
        chosen_offsets: shape [T] array of selected offsets (seconds).
    """
    offsets = np.asarray(offsets, dtype=float)
    s = np.asarray(scores, dtype=float)
    if offsets.ndim != 1:
        raise ValueError("offsets must be 1D")
    if s.ndim != 2:
        raise ValueError("scores must be 2D")
    if s.shape[1] != offsets.shape[0]:
        raise ValueError("scores.shape[1] must match offsets.shape[0]")
    smoothness_lambda = float(smoothness_lambda)
    if smoothness_lambda < 0:
        raise ValueError("smoothness_lambda must be >= 0")

    if s.shape[0] == 0:
        return np.zeros((0,), dtype=float)

    if smoothness_lambda == 0.0:
        # Tie-break argmax: if scores are flat, prefer offsets near 0.
        best = []
        for t in range(s.shape[0]):
            row = s[t]
            if not np.any(np.isfinite(row)):
                best.append(0.0)
                continue
            m = float(np.nanmax(row))
            idxs = np.flatnonzero(np.isfinite(row) & (row == m))
            if idxs.size == 0:
                best.append(0.0)
                continue
            j = int(idxs[np.argmin(np.abs(offsets[idxs]))])
            best.append(float(offsets[j]))
        return np.asarray(best, dtype=float)

    # Normalize for stability across files/keypoints.
    s_norm = normalize_scores_per_keypoint(s)

    T, M = s_norm.shape
    dp = np.full((T, M), -np.inf, dtype=float)
    back = np.full((T, M), -1, dtype=int)

    dp[0] = s_norm[0]
    for t in range(1, T):
        for j in range(M):
            trans = dp[t - 1] - smoothness_lambda * np.abs(offsets[j] - offsets)
            if not np.any(np.isfinite(trans)):
                k = 0
                dp[t, j] = s_norm[t, j]
                back[t, j] = k
                continue

            m = float(np.max(trans))
            # Near-tie handling: avoid systematic bias to first index.
            tol = 1e-12
            idxs = np.flatnonzero(np.isfinite(trans) & (trans >= m - tol))
            if idxs.size == 1:
                k = int(idxs[0])
            else:
                # Primary: choose previous offset closest to current offset (smoothness).
                # Secondary: choose offset closest to 0 (avoid extreme offsets when ambiguous).
                d1 = np.abs(offsets[idxs] - offsets[j])
                m1 = float(np.min(d1))
                idxs2 = idxs[d1 <= m1 + 1e-15]
                if idxs2.size == 1:
                    k = int(idxs2[0])
                else:
                    k = int(idxs2[np.argmin(np.abs(offsets[idxs2]))])

            dp[t, j] = s_norm[t, j] + float(trans[k])
            back[t, j] = int(k)

    last = dp[T - 1]
    if not np.any(np.isfinite(last)):
        j = int(np.argmin(np.abs(offsets)))
    else:
        m = float(np.max(last))
        tol = 1e-12
        idxs = np.flatnonzero(np.isfinite(last) & (last >= m - tol))
        if idxs.size == 1:
            j = int(idxs[0])
        else:
            j = int(idxs[np.argmin(np.abs(offsets[idxs]))])
    path = np.empty((T,), dtype=int)
    path[T - 1] = j
    for t in range(T - 1, 0, -1):
        j = int(back[t, j])
        if j < 0:
            j = 0
        path[t - 1] = j

    return offsets[path]


def otsu_threshold(values: np.ndarray, n_bins: int) -> float | None:
    """Compute Otsu threshold for 1D values using a histogram.

    Returns None if the threshold cannot be computed (e.g., degenerate input).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None

    hist, bin_edges = np.histogram(v, bins=int(n_bins))
    if hist.size < 2 or np.all(hist == 0):
        return None

    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    w1 = np.cumsum(hist).astype(float)
    w2 = np.cumsum(hist[::-1])[::-1].astype(float)
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return None

    mu1 = np.cumsum(hist * bin_mids) / np.maximum(w1, 1e-12)
    mu2 = (np.cumsum((hist * bin_mids)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]

    sigma_b2 = w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2
    if sigma_b2.size == 0 or not np.any(np.isfinite(sigma_b2)):
        return None
    idx = int(np.nanargmax(sigma_b2))
    if idx < 0 or idx >= bin_mids.size:
        return None
    return float(bin_mids[idx])


def clamp_start(start: int, n_samples: int, clip: int) -> int:
    """Clamp a window start index so [start, start+clip) stays within [0, n_samples)."""
    n_samples = int(n_samples)
    clip = int(clip)
    if clip <= 0:
        raise ValueError("clip must be > 0")
    if n_samples <= clip:
        return 0
    return max(0, min(int(start), n_samples - clip))

def remove_duplicate_keypoints(keypoints_0, keypoints_1):
    """
    When keypoints are generated, some timestamp values are repeated to avoid having timestamps beyond clip boundaries. This function removes duplicate keypoints at the beginning and end of the proposed keypoints.

    Args:
    keypoints_0 (np.ndarray): A 1D NumPy array of keypoints from the first set.
    keypoints_1 (np.ndarray): A 1D NumPy array of keypoints from the second set.

    Returns:
    tuple[np.ndarray, np.ndarray]: The cleaned keypoint arrays with duplicates at the boundaries removed.
    """    
    keypoints_min = min(np.amin(keypoints_0), np.amin(keypoints_1))
    keypoints_max = max(np.amax(keypoints_0), np.amax(keypoints_1))
        
    k0start = np.nonzero(keypoints_0 == keypoints_min)[0]
    k0start = np.amax(k0start) if len(k0start) else 0
    k1start = np.nonzero(keypoints_1 == keypoints_min)[0]
    k1start = np.amax(k1start) if len(k1start) else 0
    
    k0end = np.nonzero(keypoints_0 == keypoints_max)[0]
    k0end = np.amin(k0end) if len(k0end) else len(keypoints_0)+1
    k1end = np.nonzero(keypoints_1 == keypoints_max)[0]
    k1end = np.amin(k1end) if len(k1end) else len(keypoints_1)+1
    
    start_idx = max(k0start, k1start)
    end_idx = min(k0end,k1end)
    keypoints_0 = keypoints_0[start_idx:end_idx]
    keypoints_1 = keypoints_1[start_idx:end_idx]
    return keypoints_0, keypoints_1

def generate_candidate_keypoints(keypoints_0, max_allowed_error, duration, n_delays_to_try):
    """
    Generate candidate keypoint sets, assuming desynchronization consists of an offset + linear drift

    Args:
    keypoints_0 (np.ndarray): A 1D NumPy array containing keypoints from channel 0.
    max_allowed_error (float): The maximum allowable difference in seconds between keypoints_0 and generated keypoints.
    duration (float): The duration of the audio file in seconds.
    n_delays_to_try (int): The number of candidate keypoint sets to generate.

    Returns:
    list[np.ndarray]: A list of NumPy arrays, each representing a modified keypoint set with 
                      the same shape as keypoints_0 and within the allowable error.
    """
    
    candidate_offsets = np.linspace(-max_allowed_error, max_allowed_error, num=n_delays_to_try)
    max_slope = max_allowed_error / duration
    candidate_slopes = np.linspace(1-max_slope, 1+max_slope, num=n_delays_to_try)
    
    candidate_keypoints = []
    for offset in candidate_offsets:
        for slope in candidate_slopes:
            keypoints_1 = slope*keypoints_0 + offset
            keypoints_1 = np.maximum(keypoints_1, 0)
            keypoints_1 = np.minimum(keypoints_1, duration)
            observed_error = np.amax(np.abs(keypoints_1 - keypoints_0))
            if observed_error <= max_allowed_error:
                candidate_keypoints.append(keypoints_1)
    
    return candidate_keypoints


def generate_candidate_keypoints_offset_only(keypoints_0, max_allowed_error, duration, offset_step):
    raise RuntimeError(
        "generate_candidate_keypoints_offset_only was removed: unused in the current baselines"
    )

def load_audio(audio_fp, sr=None):
    """
    Load an audio file and optionally resample it to a specified sample rate.

    Args:
    audio_fp (str): File path to the audio file.
    sr (int, optional): Desired sample rate. If None, the original sample rate is used.

    Returns:
    tuple[torch.Tensor, int]: A tuple containing the loaded audio as a Torch tensor and the sample rate.
    """
    audio, orig_sr = torchaudio.load(audio_fp)
    if sr is None:
        return audio, orig_sr
    else:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
        return audio, sr

def pad_to_dur(audio, sr, dur_sec):
    """
    Pad or crop an audio tensor to match a specified duration.

    Args:
    audio (torch.Tensor): Input audio tensor with shape [..., N], where N is the number of samples.
    sr (int): Sample rate of the audio.
    dur_sec (float): Desired duration of the output audio in seconds.

    Returns:
    torch.Tensor: Audio tensor padded or cropped to have a duration of sr * dur_sec samples.
    """
    desired_dur_samples = int(sr*dur_sec)
    audio_dur_samples = audio.size(-1)
    pad = desired_dur_samples - audio_dur_samples
    if pad>0:
        audio = nn.functional.pad(audio, (0,pad))
    audio = audio[...,:desired_dur_samples]
    return audio