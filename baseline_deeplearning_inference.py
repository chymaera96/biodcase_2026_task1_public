"""Baseline system: pairwise alignment scoring + offset-grid search.

Usage:
python baseline_deeplearning_inference.py --output-dir=/path/to/desired/output/dir \
    --inference-dir=/path/to/audio/folder --pretrained-fp=/path/to/model.pt --config=configs/aru.yaml
"""

import argparse
import json
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd
import torch
from models import BEATsEncoderAndMLP
from tqdm import tqdm
from utils import (
    generate_candidate_keypoints,
    make_offset_grid,
    load_audio,
    load_hyperparams,
    otsu_threshold,
    pad_to_dur,
    require_hyperparams,
    viterbi_smooth_offsets,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _rms_db(x: torch.Tensor) -> float:
    x = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(x * x) + 1e-12)
    return 20.0 * float(torch.log10(rms + 1e-12).item())


def _load_saved_silence_threshold(pretrained_fp: str) -> float | None:
    d = os.path.dirname(os.path.abspath(pretrained_fp))
    fp = os.path.join(d, "silence_threshold.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        thr = float(meta["threshold_db"])
        return thr
    except Exception:
        return None


def _score_mode_to_score(logits: torch.Tensor, score_mode: str) -> torch.Tensor:
    if score_mode == "count_pos":
        return (logits > 0).to(torch.float32)
    if score_mode == "sum_logits":
        return logits
    if score_mode == "sum_prob":
        return torch.sigmoid(logits)
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _score_offsets_for_keypoint(
    audio: torch.Tensor,
    sr: int,
    k0_sec: float,
    offsets_sec: np.ndarray,
    window_size: float,
    audio_dur: float,
    model: torch.nn.Module,
    score_mode: str,
    batch_size: int,
) -> np.ndarray:
    """Score each candidate offset for a single keypoint."""
    offsets_sec = np.asarray(offsets_sec, dtype=float)
    scores = np.full((len(offsets_sec),), -np.inf, dtype=float)

    k0_sec = float(k0_sec)
    win = float(window_size)
    if not (0.0 <= k0_sec and (k0_sec + win) <= audio_dur):
        return scores

    k0 = int(round(sr * k0_sec))
    dur = int(round(sr * win))
    audio_0 = audio[0, k0 : k0 + dur]
    audio_0 = pad_to_dur(audio_0, sr, win)

    valid = []
    audio_1s = []
    for j, off in enumerate(offsets_sec):
        k1_sec = k0_sec + float(off)
        if not (0.0 <= k1_sec and (k1_sec + win) <= audio_dur):
            continue
        k1 = int(round(sr * k1_sec))
        audio_1 = audio[1, k1 : k1 + dur]
        audio_1 = pad_to_dur(audio_1, sr, win)
        valid.append(j)
        audio_1s.append(audio_1)

    if not valid:
        return scores

    audio_1s = torch.stack(audio_1s, dim=0)

    # Performance: avoid recomputing the channel-0 embedding for every offset candidate.
    # The model is BEATsEncoderAndMLP, which exposes `embed()` and `head`.
    all_scores = []
    with torch.inference_mode():
        if hasattr(model, "embed") and hasattr(model, "head"):
            a0 = audio_0.unsqueeze(0).to(device)
            e0 = model.embed(a0)  # [1, D]
            for start in range(0, audio_1s.shape[0], int(batch_size)):
                end = min(start + int(batch_size), audio_1s.shape[0])
                a1 = audio_1s[start:end].to(device)
                e1 = model.embed(a1)  # [B, D]
                e0_rep = e0.expand(e1.shape[0], -1)
                if hasattr(model, "pair_features"):
                    feats = model.pair_features(e0_rep, e1)
                else:
                    feats = torch.cat([e0_rep, e1, torch.abs(e0_rep - e1)], dim=-1)
                logits = model.head(feats).squeeze(-1)
                sc = _score_mode_to_score(logits, score_mode=score_mode)
                all_scores.append(sc.detach().cpu())
        else:
            audio_0s = audio_0.unsqueeze(0).expand(audio_1s.shape[0], -1)
            for start in range(0, audio_1s.shape[0], int(batch_size)):
                end = min(start + int(batch_size), audio_1s.shape[0])
                logits = model(audio_0s[start:end].to(device), audio_1s[start:end].to(device))
                sc = _score_mode_to_score(logits, score_mode=score_mode)
                all_scores.append(sc.detach().cpu())
    all_scores = torch.cat(all_scores, dim=0).numpy().astype(float)

    for idx, j in enumerate(valid):
        scores[j] = float(all_scores[idx])
    return scores


def _score_alignment_candidate(
    *,
    audio: torch.Tensor,
    sr: int,
    audio_dur: float,
    keypoints_0: np.ndarray,
    keypoints_1: np.ndarray,
    idxs_non_silent: np.ndarray,
    idxs_all_for_mean: np.ndarray,
    window_size: float,
    model: torch.nn.Module,
    score_mode: str,
    batch_size: int,
) -> float:
    """Aggregate model score over paired windows for one candidate mapping.

    - Uses only idxs_non_silent for model evaluation.
    - Averages over idxs_all_for_mean (silent windows contribute 0 if included).
    """

    idxs_non_silent = np.asarray(idxs_non_silent, dtype=int)
    idxs_all_for_mean = np.asarray(idxs_all_for_mean, dtype=int)
    if idxs_all_for_mean.size == 0:
        return -np.inf

    total = float(idxs_all_for_mean.size)
    if idxs_non_silent.size == 0:
        # All-silent under neutral policy, or nothing usable.
        return 0.0 if total > 0 else -np.inf

    win = float(window_size)
    dur = int(round(float(sr) * win))
    if dur <= 0:
        raise ValueError("window_size must be > 0")

    # Build paired window batches in the candidate's keypoint order.
    a0_list = []
    a1_list = []
    for i in idxs_non_silent:
        k0_sec = float(keypoints_0[i])
        k1_sec = float(keypoints_1[i])
        # Enforce full windows; skip any that would fall out of bounds.
        if not (0.0 <= k0_sec and (k0_sec + win) <= audio_dur):
            continue
        if not (0.0 <= k1_sec and (k1_sec + win) <= audio_dur):
            continue

        k0 = int(round(float(sr) * k0_sec))
        k1 = int(round(float(sr) * k1_sec))
        w0 = pad_to_dur(audio[0, k0 : k0 + dur], sr, win)
        w1 = pad_to_dur(audio[1, k1 : k1 + dur], sr, win)
        a0_list.append(w0)
        a1_list.append(w1)

    if not a0_list:
        return -np.inf

    a0 = torch.stack(a0_list, dim=0)
    a1 = torch.stack(a1_list, dim=0)

    sum_score = 0.0
    n_used = 0
    with torch.inference_mode():
        if hasattr(model, "embed") and hasattr(model, "head"):
            for start in range(0, a0.shape[0], int(batch_size)):
                end = min(start + int(batch_size), a0.shape[0])
                e0 = model.embed(a0[start:end].to(device))
                e1 = model.embed(a1[start:end].to(device))
                if hasattr(model, "pair_features"):
                    feats = model.pair_features(e0, e1)
                else:
                    feats = torch.cat([e0, e1, torch.abs(e0 - e1)], dim=-1)
                logits = model.head(feats).squeeze(-1)
                sc = _score_mode_to_score(logits, score_mode=score_mode)
                sum_score += float(sc.detach().float().sum().cpu().item())
                n_used += int(sc.numel())
        else:
            for start in range(0, a0.shape[0], int(batch_size)):
                end = min(start + int(batch_size), a0.shape[0])
                logits = model(a0[start:end].to(device), a1[start:end].to(device))
                sc = _score_mode_to_score(logits, score_mode=score_mode)
                sum_score += float(sc.detach().float().sum().cpu().item())
                n_used += int(sc.numel())

    if n_used == 0:
        return -np.inf

    # Mean across all-for-mean indices; silent ones contribute 0 via not being in idxs_non_silent.
    return float(sum_score) / float(total)


def main():
    args = parse_args()
    args.hyperparam = load_hyperparams(args.config)
    hp = args.hyperparam

    keys = [
        "max_error",
        "window_size",
        "time_bins",
        "score_mode",
        "silence_mode",
        "silence_policy",
        "silence_hist_bins",
        "silence_percentile_fallback",
        "infer_batch_size",
        "global_affine_search",
    ]
    if bool(hp.global_affine_search):
        keys.append("global_affine_n_per_axis")
    else:
        keys.extend(
            [
                "keypoint_stride",
                "offset_step",
                "dp_lambda",
                "disable_refinement",
                "refine_half_width_sec",
            ]
        )
    require_hyperparams(hp, keys, config_fp=args.config)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    predictions = {"Filename": [], "Time Channel 0": [], "Time Channel 1": []}
    audio_fps = sorted(glob(os.path.join(args.inference_dir, "*")))

    # Load pretrained model
    model = BEATsEncoderAndMLP(args.beats_fp, time_bins=int(hp.time_bins)).to(device)
    model.load_state_dict(torch.load(args.pretrained_fp, weights_only=True))
    model.eval()

    saved_thr_db = None
    if str(hp.silence_mode) != "off":
        saved_thr_db = _load_saved_silence_threshold(args.pretrained_fp)
        if saved_thr_db is not None:
            print(f"Loaded silence threshold from model dir: threshold_db={saved_thr_db:.2f}")

    # Iterate through audio files
    for audio_fp in tqdm(audio_fps):
        audio, sr = load_audio(audio_fp, sr=16000)
        audio_dur = librosa.get_duration(path=audio_fp)

        # Integer-second keypoints in channel 0.
        n_keypoints = int(np.floor(audio_dur))
        keypoints_0 = np.arange(0, n_keypoints, dtype=float)

        # Silence gating on channel-0 windows.
        silent = np.zeros((len(keypoints_0),), dtype=bool)
        threshold_db = None
        if str(hp.silence_mode) != "off" and len(keypoints_0) > 0:
            if saved_thr_db is not None:
                threshold_db = float(saved_thr_db)
            else:
                # Per-file auto threshold.
                rms_vals = []
                dur = int(round(sr * float(hp.window_size)))
                for k0_sec in keypoints_0:
                    k0 = int(round(sr * float(k0_sec)))
                    w0 = pad_to_dur(audio[0, k0 : k0 + dur], sr, float(hp.window_size))
                    rms_vals.append(_rms_db(w0))
                rms_vals = np.asarray(rms_vals, dtype=float)
                finite = rms_vals[np.isfinite(rms_vals)]
                fallback = float(np.percentile(finite, float(hp.silence_percentile_fallback))) if finite.size else 0.0
                thr = otsu_threshold(rms_vals, n_bins=int(hp.silence_hist_bins))
                if thr is None:
                    thr = fallback
                else:
                    # Keep gating lenient: never more aggressive than the fallback percentile.
                    thr = min(float(thr), float(fallback))
                threshold_db = float(thr)

            # Mark silent windows.
            dur = int(round(sr * float(hp.window_size)))
            for i, k0_sec in enumerate(keypoints_0):
                k0 = int(round(sr * float(k0_sec)))
                w0 = pad_to_dur(audio[0, k0 : k0 + dur], sr, float(hp.window_size))
                silent[i] = _rms_db(w0) < float(threshold_db)

        # Global affine search mode (legacy-style): score candidate affine mappings
        # and pick the best one for the entire file.
        if bool(hp.global_affine_search):
            keypoints_0_arr = np.asarray(keypoints_0, dtype=float)
            if keypoints_0_arr.size == 0:
                continue
            n_try = int(hp.global_affine_n_per_axis)
            if n_try < 1:
                raise ValueError("global_affine_n_per_axis must be >= 1")

            cand_keypoints_1 = generate_candidate_keypoints(
                keypoints_0_arr,
                float(hp.max_error),
                float(audio_dur),
                int(n_try),
            )
            if not cand_keypoints_1:
                # Fall back to identity mapping.
                best_keypoints_1 = keypoints_0_arr.copy()
            else:
                best_score = -np.inf
                best_keypoints_1 = None

                # Determine which keypoints participate in scoring.
                idxs_all = np.arange(len(keypoints_0_arr), dtype=int)
                if str(hp.silence_mode) != "off" and str(hp.silence_policy) in ("skip", "interp"):
                    idxs_non_silent = idxs_all[~silent]
                    idxs_all_for_mean = idxs_non_silent
                elif str(hp.silence_mode) != "off" and str(hp.silence_policy) == "neutral":
                    idxs_non_silent = idxs_all[~silent]
                    idxs_all_for_mean = idxs_all
                else:
                    idxs_non_silent = idxs_all
                    idxs_all_for_mean = idxs_all

                if len(idxs_all_for_mean) == 0:
                    fn = os.path.basename(audio_fp)
                    predictions["Filename"].extend([fn for _ in keypoints_0_arr])
                    predictions["Time Channel 0"].extend(list(map(float, keypoints_0_arr)))
                    predictions["Time Channel 1"].extend(list(map(float, keypoints_0_arr)))
                    continue

                # Precompute channel-0 embeddings once (major speedup).
                use_fast_path = hasattr(model, "embed") and hasattr(model, "head")
                idxs_k0_ok: np.ndarray | None = None
                e0_all: torch.Tensor | None = None
                if use_fast_path:
                    idxs_k0 = []
                    a0_list = []
                    win = float(hp.window_size)
                    dur = int(round(float(sr) * win))
                    for i in idxs_non_silent:
                        k0_sec = float(keypoints_0_arr[i])
                        if not (0.0 <= k0_sec and (k0_sec + win) <= float(audio_dur)):
                            continue
                        k0 = int(round(float(sr) * k0_sec))
                        w0 = pad_to_dur(audio[0, k0 : k0 + dur], int(sr), win)
                        a0_list.append(w0)
                        idxs_k0.append(int(i))

                    if a0_list:
                        a0 = torch.stack(a0_list, dim=0)
                        e0_chunks = []
                        with torch.inference_mode():
                            for start in range(0, a0.shape[0], int(hp.infer_batch_size)):
                                end = min(start + int(hp.infer_batch_size), a0.shape[0])
                                e0_chunks.append(model.embed(a0[start:end].to(device)))
                        e0_all = torch.cat(e0_chunks, dim=0)
                        idxs_k0_ok = np.asarray(idxs_k0, dtype=int)
                    else:
                        idxs_k0_ok = np.asarray([], dtype=int)
                        e0_all = None

                for kp1 in cand_keypoints_1:
                    kp1 = np.asarray(kp1, dtype=float)
                    if use_fast_path and idxs_k0_ok is not None and e0_all is not None:
                        win = float(hp.window_size)
                        dur = int(round(float(sr) * win))
                        a1_list = []
                        e0_list = []
                        for pos, i in enumerate(idxs_k0_ok):
                            k1_sec = float(kp1[i])
                            if not (0.0 <= k1_sec and (k1_sec + win) <= float(audio_dur)):
                                continue
                            k1 = int(round(float(sr) * k1_sec))
                            w1 = pad_to_dur(audio[1, k1 : k1 + dur], int(sr), win)
                            a1_list.append(w1)
                            e0_list.append(e0_all[pos])

                        if not a1_list:
                            score = -np.inf
                        else:
                            a1 = torch.stack(a1_list, dim=0)
                            e0 = torch.stack(e0_list, dim=0)
                            sum_score = 0.0
                            with torch.inference_mode():
                                for start in range(0, a1.shape[0], int(hp.infer_batch_size)):
                                    end = min(start + int(hp.infer_batch_size), a1.shape[0])
                                    e1 = model.embed(a1[start:end].to(device))
                                    e0_b = e0[start:end].to(device)
                                    if hasattr(model, "pair_features"):
                                        feats = model.pair_features(e0_b, e1)
                                    else:
                                        feats = torch.cat([e0_b, e1, torch.abs(e0_b - e1)], dim=-1)
                                    logits = model.head(feats).squeeze(-1)
                                    sc = _score_mode_to_score(logits, score_mode=str(hp.score_mode))
                                    sum_score += float(sc.detach().float().sum().cpu().item())

                            score = float(sum_score) / float(len(idxs_all_for_mean))
                    else:
                        score = _score_alignment_candidate(
                            audio=audio,
                            sr=int(sr),
                            audio_dur=float(audio_dur),
                            keypoints_0=keypoints_0_arr,
                            keypoints_1=kp1,
                            idxs_non_silent=idxs_non_silent,
                            idxs_all_for_mean=idxs_all_for_mean,
                            window_size=float(hp.window_size),
                            model=model,
                            score_mode=str(hp.score_mode),
                            batch_size=int(hp.infer_batch_size),
                        )

                    if score > best_score:
                        best_score = float(score)
                        best_keypoints_1 = kp1

                if best_keypoints_1 is None:
                    best_keypoints_1 = keypoints_0_arr.copy()

            fn = os.path.basename(audio_fp)
            predictions["Filename"].extend([fn for _ in keypoints_0_arr])
            predictions["Time Channel 0"].extend(list(map(float, keypoints_0_arr)))
            predictions["Time Channel 1"].extend(list(map(float, best_keypoints_1)))
            continue

        # Per-keypoint offset search (handles non-affine drift + steps).
        fine_step = float(hp.offset_step)
        if fine_step <= 0:
            raise ValueError("offset_step must be > 0")
        coarse_step = max(0.05, 5.0 * fine_step)
        offsets_coarse = make_offset_grid(float(hp.max_error), coarse_step)

        # Optionally evaluate on a subset for speed, then interpolate offsets.
        if int(hp.keypoint_stride) < 1:
            raise ValueError("keypoint_stride must be >= 1")
        eval_idx = np.arange(0, len(keypoints_0), int(hp.keypoint_stride), dtype=int)
        if str(hp.silence_mode) != "off" and str(hp.silence_policy) == "interp":
            eval_idx = eval_idx[~silent[eval_idx]]
        keypoints_0_eval = keypoints_0[eval_idx]

        coarse_scores = []
        for k0 in keypoints_0_eval:
            s = _score_offsets_for_keypoint(
                audio,
                sr,
                float(k0),
                offsets_coarse,
                float(hp.window_size),
                float(audio_dur),
                model,
                score_mode=str(hp.score_mode),
                batch_size=int(hp.infer_batch_size),
            )
            coarse_scores.append(s)
        coarse_scores = np.stack(coarse_scores, axis=0) if len(coarse_scores) else np.zeros((0, len(offsets_coarse)))

        offsets_path_eval = viterbi_smooth_offsets(offsets_coarse, coarse_scores, smoothness_lambda=float(hp.dp_lambda))

        # Interpolate coarse offsets to all keypoints.
        if len(keypoints_0_eval) >= 2:
            offsets_path_coarse = np.interp(keypoints_0, keypoints_0_eval, offsets_path_eval)
        elif len(keypoints_0_eval) == 1:
            offsets_path_coarse = np.full_like(keypoints_0, float(offsets_path_eval[0]))
        else:
            offsets_path_coarse = np.zeros_like(keypoints_0)

        # Refinement per keypoint (optional).
        best_keypoints_1 = np.empty_like(keypoints_0)

        # Default refinement radius matches previous behavior unless overridden.
        refine_half_width = (
            float(hp.refine_half_width_sec)
            if hp.refine_half_width_sec is not None
            else float(coarse_step)
        )

        for i, k0 in enumerate(keypoints_0):
            # Under interp, silent keypoints should not be re-scored (they contain
            # little/no information). Instead, inherit the interpolated coarse offset.
            if str(hp.silence_mode) != "off" and str(hp.silence_policy) == "interp" and bool(silent[i]):
                best_keypoints_1[i] = float(k0) + float(offsets_path_coarse[i])
                continue

            center = float(offsets_path_coarse[i])

            # Option 1: disable refinement entirely and use the coarse decoded path.
            if bool(hp.disable_refinement):
                best_keypoints_1[i] = float(k0) + center
                continue

            # Option 2: refine in a configurable band around the coarse path.
            lo = max(-float(hp.max_error), center - refine_half_width)
            hi = min(float(hp.max_error), center + refine_half_width)
            offsets_fine = np.arange(lo, hi + 0.5 * fine_step, fine_step, dtype=float)

            if offsets_fine.size == 0:
                best_keypoints_1[i] = float(k0) + center
                continue

            scores_fine = _score_offsets_for_keypoint(
                audio,
                sr,
                float(k0),
                offsets_fine,
                float(hp.window_size),
                float(audio_dur),
                model,
                score_mode=str(hp.score_mode),
                batch_size=int(hp.infer_batch_size),
            )
            j = int(np.argmax(scores_fine))
            if not np.isfinite(scores_fine[j]):
                best_keypoints_1[i] = float(k0) + center
            else:
                best_keypoints_1[i] = float(k0) + float(offsets_fine[j])

        # Save best keypoint proposal for the file
        fn = os.path.basename(audio_fp)
        keypoints_0 = list(keypoints_0)
        keypoints_1 = list(best_keypoints_1)
        predictions["Filename"].extend([fn for _ in keypoints_0])
        predictions["Time Channel 0"].extend(keypoints_0)
        predictions["Time Channel 1"].extend(keypoints_1)

    # Save all predictions
    predictions = pd.DataFrame(predictions)
    output_fp = os.path.join(args.output_dir, "predictions.csv")
    predictions.to_csv(output_fp, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    parser.add_argument("--inference-dir", type=str, required=True, help="path to audio folder to make predictions for")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config containing hyperparameters (loaded into args.hyperparam.*)",
    )
    parser.add_argument("--pretrained-fp", type=str, required=True, help="path to model weights if using pretraining")
    parser.add_argument(
        "--beats-fp",
        type=str,
        default="BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        help="Path to beats checkpoint, can be obtained from https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
