"""Baseline system: GCC-PHAT delay estimation.

Estimates the per-keypoint delay between channel 0 and channel 1 using GCC-PHAT
(phase transform) within +/- max_error seconds.

Outputs predictions.csv with columns:
  Filename, Time Channel 0, Time Channel 1

Usage:
    python baseline_gccphat.py --output-dir=/path/to/out --inference-dir=/path/to/audio --config=configs/aru.yaml
"""

import argparse
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_hyperparams, make_offset_grid, require_hyperparams, viterbi_smooth_offsets


def _next_pow_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _extract_padded(audio_1d: np.ndarray, start: int, length: int) -> np.ndarray:
    """Extract a fixed-length segment with zero padding."""
    start = int(start)
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float64)

    end = start + length
    x = np.zeros((length,), dtype=np.float64)

    src_start = max(0, start)
    src_end = min(int(audio_1d.shape[-1]), end)
    if src_end <= src_start:
        return x

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    x[dst_start:dst_end] = audio_1d[src_start:src_end].astype(np.float64, copy=False)
    return x


def _gcc_phat_delay_sec(
    sig: np.ndarray,
    ref: np.ndarray,
    sr: int,
    max_tau_sec: float,
    eps: float = 1e-12,
    apply_hann: bool = True,
) -> float:
    """Return delay (seconds) of sig relative to ref using GCC-PHAT.

    Convention: positive delay means `sig` occurs later than `ref`.

    This function uses zero-padded FFT-based correlation and then searches only
    within +/- max_tau_sec.
    """

    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if sig.size == 0 or ref.size == 0:
        return 0.0

    # Remove DC (helps PHAT stability)
    sig = sig - float(np.mean(sig))
    ref = ref - float(np.mean(ref))

    if apply_hann and sig.size > 1:
        w = np.hanning(sig.size)
        sig = sig * w
        ref = ref * w

    # Zero-pad enough for linear correlation.
    n = int(sig.size + ref.size)
    nfft = _next_pow_2(n)

    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)
    # Cross-spectrum for r_ref,sig so that a delayed `sig` yields positive shift.
    R = REF * np.conj(SIG)
    R = R / (np.abs(R) + float(eps))

    cc = np.fft.irfft(R, n=nfft)

    max_shift = int(nfft // 2)
    max_tau_samp = int(round(float(max_tau_sec) * float(sr)))
    if max_tau_samp >= 0:
        max_shift = min(max_shift, max_tau_samp)

    if max_shift <= 0:
        return 0.0

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    return float(shift) / float(sr)


def _smooth_offsets_viterbi(offsets_pred: np.ndarray, max_error: float, step: float, lam: float) -> np.ndarray:
    """Optional temporal smoothing of predicted offsets using Viterbi."""
    offsets_pred = np.asarray(offsets_pred, dtype=float)
    if offsets_pred.size == 0:
        return offsets_pred

    grid = make_offset_grid(float(max_error), float(step))
    grid = np.asarray(grid, dtype=float)
    if grid.size == 0:
        return offsets_pred

    # Score shape: [T, M]
    # Use an L1-shaped score: higher is better.
    scores = -np.abs(offsets_pred[:, None] - grid[None, :]).astype(np.float32)
    smoothed = viterbi_smooth_offsets(grid, scores, float(lam))
    return np.asarray(smoothed, dtype=float)


def main() -> None:
    args = parse_args()
    hp = load_hyperparams(args.config)
    require_hyperparams(
        hp,
        [
            "max_error",
            "gccphat_window_size",
            "gccphat_smooth_lambda",
            "gccphat_smooth_step",
            "gccphat_no_hann",
        ],
        config_fp=args.config,
    )

    max_error = float(hp.max_error)
    window_size = float(hp.gccphat_window_size)
    smooth_lambda = float(hp.gccphat_smooth_lambda)
    smooth_step = float(hp.gccphat_smooth_step)
    no_hann = bool(hp.gccphat_no_hann)

    if max_error <= 0:
        raise ValueError("max_error must be > 0")
    if window_size <= 0:
        raise ValueError("gccphat_window_size must be > 0")
    if smooth_step <= 0:
        raise ValueError("gccphat_smooth_step must be > 0")
    if smooth_lambda < 0:
        raise ValueError("gccphat_smooth_lambda must be >= 0")

    os.makedirs(args.output_dir, exist_ok=True)

    predictions = {"Filename": [], "Time Channel 0": [], "Time Channel 1": []}
    audio_fps = sorted(glob(os.path.join(args.inference_dir, "*")))

    for audio_fp in tqdm(audio_fps):
        audio, sr = librosa.load(audio_fp, mono=False, sr=None)
        audio_dur = float(librosa.get_duration(path=audio_fp))

        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected 2-channel audio shaped (2, n_samples), got shape {audio.shape} for {audio_fp}")

        n_keypoints = int(np.floor(audio_dur))
        if n_keypoints <= 0:
            continue

        keypoints_0 = np.arange(0, n_keypoints, dtype=float)
        win = int(round(float(sr) * float(window_size)))
        if win <= 0:
            raise ValueError("gccphat_window_size too small")

        delays = np.zeros((n_keypoints,), dtype=float)
        for i, k0 in enumerate(keypoints_0):
            start = int(round(float(k0) * float(sr)))
            x0 = _extract_padded(audio[0], start, win)
            x1 = _extract_padded(audio[1], start, win)

            # Delay of channel 1 relative to channel 0.
            tau = _gcc_phat_delay_sec(x1, x0, sr=sr, max_tau_sec=float(max_error), apply_hann=not no_hann)
            delays[i] = float(np.clip(tau, -float(max_error), float(max_error)))

        if smooth_lambda > 0:
            delays = _smooth_offsets_viterbi(delays, max_error=float(max_error), step=float(smooth_step), lam=float(smooth_lambda))

        keypoints_1 = keypoints_0 + delays

        fn = os.path.basename(audio_fp)
        predictions["Filename"].extend([fn] * int(keypoints_0.size))
        predictions["Time Channel 0"].extend(list(map(float, keypoints_0)))
        predictions["Time Channel 1"].extend(list(map(float, keypoints_1)))

    df = pd.DataFrame(predictions)
    out_fp = os.path.join(args.output_dir, "predictions.csv")
    df.to_csv(out_fp, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    p.add_argument("--inference-dir", type=str, required=True, help="path to audio folder to make predictions for")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config containing hyperparameters (loaded via utils.load_hyperparams)",
    )
    args = p.parse_args()
    return args


if __name__ == "__main__":
    main()
