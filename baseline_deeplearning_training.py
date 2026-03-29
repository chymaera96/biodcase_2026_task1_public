"""
Train baseline deep learning system
Usage:
python baseline_deeplearning_training.py --output-dir=/path/to/desired/output/dir \
    --train-dir=/path/to/train/dir --val-dir=/path/to/val/dir --config=configs/aru.yaml
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from models import BEATsEncoderAndMLP
from torch.nn import functional as F
from tqdm import tqdm
from utils import (
    clamp_start,
    load_audio,
    load_hyperparams,
    otsu_threshold,
    pad_to_dur,
    remove_duplicate_keypoints,
    require_hyperparams,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class AudioPairDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading and processing paired audio segments from a dataset.

    Attributes:
    clip_dur (int): Duration of the extracted audio clips in seconds.
    sr (int): Sample rate of the audio.
    audio (list): A list of tuples containing paired audio segments (audio_0, audio_1).
    rng (np.random.Generator): A random number generator for dataset operations.
    """

    def __init__(
        self,
        data_dir: str,
        window_size_sec: float = 1.0,
        max_error_sec: float = 0.5,
        min_neg_offset_sec: float = 0.05,
        neg_prob: float = 0.5,
        deterministic: bool = False,
        deterministic_seed: int = 0,
    ):
        """
        Initialize the AudioPairDataset by loading audio files and their corresponding annotations.

        Args:
        data_dir (str): Path to the directory containing the 'annotations.csv' file and an 'audio' subdirectory.
        """
        # Read annotations
        annotations_fp = os.path.join(data_dir, "annotations.csv")
        annotations = pd.read_csv(annotations_fp)
        print(f"Loading Audio from {data_dir}")

        if window_size_sec <= 0:
            raise ValueError("window_size_sec must be > 0")

        if max_error_sec <= 0:
            raise ValueError("max_error_sec must be > 0")
        if min_neg_offset_sec < 0:
            raise ValueError("min_neg_offset_sec must be >= 0")
        if min_neg_offset_sec > max_error_sec:
            raise ValueError("min_neg_offset_sec must be <= max_error_sec")
        if not (0.0 <= neg_prob <= 1.0):
            raise ValueError("neg_prob must be in [0, 1]")

        self.clip_dur = float(window_size_sec)
        self.max_error_sec = float(max_error_sec)
        self.min_neg_offset_sec = float(min_neg_offset_sec)
        self.neg_prob = float(neg_prob)
        self.sr = 16000
        self.examples = []
        self.anchor_rms_db = []

        # Mutable training knobs (optionally adjusted by the training loop).
        self.neg_curriculum_min_abs_sec = float(min_neg_offset_sec)

        # Deterministic mode: make all sampling depend only on idx.
        # This is used to remove randomness from validation.
        self.deterministic = bool(deterministic)
        self.deterministic_seed = int(deterministic_seed)

        audio_fns = sorted(annotations["Filename"].unique())

        clip = int(round(self.sr * self.clip_dur))
        max_err = int(round(self.sr * self.max_error_sec))

        for audio_fn in tqdm(audio_fns):
            # Load audio
            audio_fp = os.path.join(data_dir, "audio", audio_fn)
            audio, sr = load_audio(audio_fp, sr=self.sr)
            n_samples = int(audio.shape[-1])

            # Load keypoints
            annotations_sub = annotations[annotations["Filename"] == audio_fn]
            keypoints_0 = annotations_sub["Time Channel 0"].values
            keypoints_1 = annotations_sub["Time Channel 1"].values

            # Remove duplicate keypoints at audio recording boundaries
            keypoints_0, keypoints_1 = remove_duplicate_keypoints(keypoints_0, keypoints_1)

            # Create examples
            for ii in range(len(keypoints_0) - 1):
                # Store context windows around the aligned keypoint so we can create
                # offset-perturbed negatives without re-loading full audio.
                ctx = self.clip_dur + 2.0 * self.max_error_sec
                k0 = float(keypoints_0[ii])
                k1 = float(keypoints_1[ii])
                offset_sec = float(k1 - k0)
                if abs(offset_sec) > float(self.max_error_sec) + 1e-12:
                    # Can't represent this within the local +/- max_error context.
                    continue
                offset_samples = int(round(sr * offset_sec))

                # Anchor BOTH contexts at channel-0 keypoint time so that offsets are
                # expressed consistently as k1 = k0 + delta.
                start_0 = int(round(sr * (k0 - self.max_error_sec)))
                start_1 = int(round(sr * (k0 - self.max_error_sec)))
                end_0 = start_0 + int(ctx * sr)
                end_1 = start_1 + int(ctx * sr)

                # If we clip the context start at 0, we must remember how much was clipped;
                # otherwise the downstream indexing (which assumes the keypoint is exactly
                # max_error seconds into the context) becomes wrong and positives become noisy.
                left_clip_0 = int(max(0, -start_0))
                left_clip_1 = int(max(0, -start_1))

                start_0c = max(start_0, 0)
                start_1c = max(start_1, 0)
                end_0c = max(min(end_0, n_samples), 0)
                end_1c = max(min(end_1, n_samples), 0)

                audio_0 = audio[0, start_0c:end_0c]
                audio_1 = audio[1, start_1c:end_1c]
                audio_0 = pad_to_dur(audio_0, sr, ctx)
                audio_1 = pad_to_dur(audio_1, sr, ctx)

                # Drop examples where the left clip would make the nominal aligned window
                # start before the beginning of the stored context.
                max_err = int(round(sr * self.max_error_sec))
                if left_clip_0 > max_err or left_clip_1 > max_err:
                    continue

                # Ensure the nominal positive (true) window for channel 1 fits in context.
                start_0_in_ctx = max_err - int(left_clip_0)
                start_1_in_ctx = start_0_in_ctx + int(offset_samples)
                if start_1_in_ctx < 0 or (start_1_in_ctx + clip) > int(audio_1.numel()):
                    continue

                self.examples.append((audio_0, audio_1, left_clip_0, offset_samples))

                # RMS over the anchor (channel-0) aligned window.
                start_0 = start_0_in_ctx
                w0 = audio_0[start_0 : start_0 + clip].to(torch.float32)
                rms = torch.sqrt(torch.mean(w0 * w0) + 1e-12).item()
                rms_db = 20.0 * float(np.log10(rms + 1e-12))
                self.anchor_rms_db.append(rms_db)

        self.rng = np.random.default_rng(0)

    def _rng_for_idx(self, idx: int) -> np.random.Generator:
        if bool(self.deterministic):
            # idx-based seed makes behavior independent of dataloader workers and iteration order.
            return np.random.default_rng(int(self.deterministic_seed) + int(idx))
        return self.rng

    def apply_silence_threshold(self, threshold_db: float) -> None:
        """Drop examples whose anchor RMS dB is below threshold."""
        threshold_db = float(threshold_db)
        if len(self.examples) != len(self.anchor_rms_db):
            raise RuntimeError("anchor_rms_db length mismatch")
        keep = [db >= threshold_db for db in self.anchor_rms_db]
        before = len(self.examples)
        if before == 0:
            return
        self.examples = [ex for ex, k in zip(self.examples, keep) if k]
        self.anchor_rms_db = [db for db, k in zip(self.anchor_rms_db, keep) if k]
        after = len(self.examples)
        print(f"Silence gating: kept {after}/{before} examples (threshold_db={threshold_db:.2f})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        audio_0_ctx, audio_1_ctx, left_clip_0, offset_samples = self.examples[idx]

        rng = self._rng_for_idx(int(idx))

        # Positive: aligned; Negative: apply an offset perturbation to channel 1.
        is_negative = bool(rng.random() < self.neg_prob)

        sr = self.sr
        clip = int(round(sr * self.clip_dur))
        max_err = int(round(sr * self.max_error_sec))

        # Context is [k - max_err, k - max_err + ctx] but may have been left-clipped to 0.
        # Adjust starts so that delta=0 remains aligned in the stored context.
        start_0 = max_err - int(left_clip_0)
        end_0 = start_0 + clip

        # True (positive) offset is relative to channel-0 time.
        start_1_base = int(start_0 + int(offset_samples))

        if not is_negative:
            label = 1.0
            start_1 = clamp_start(start_1_base, int(audio_1_ctx.numel()), int(clip))
        else:
            label = 0.0
            # Resample negatives if boundary clamping would erase the intended offset.
            max_resamples = 20
            for _ in range(max_resamples):
                min_abs = float(self.neg_curriculum_min_abs_sec)
                if min_abs <= 0:
                    delta_sec = float(rng.uniform(-self.max_error_sec, self.max_error_sec))
                else:
                    sign = -1.0 if (rng.random() < 0.5) else 1.0
                    mag = float(rng.uniform(min_abs, self.max_error_sec))
                    delta_sec = sign * mag
                delta = int(round(sr * float(delta_sec)))

                start_1 = clamp_start(start_1_base + delta, int(audio_1_ctx.numel()), int(clip))
                # Enforce a minimum deviation from the true aligned window.
                realized_delta = float(start_1 - start_1_base) / float(sr)
                if abs(realized_delta) + 1e-12 >= float(self.min_neg_offset_sec):
                    break
            else:
                # If we fail to sample a clean negative, fall back to a positive.
                label = 1.0
                start_1 = clamp_start(start_1_base, int(audio_1_ctx.numel()), int(clip))

        end_1 = start_1 + clip

        audio_0 = audio_0_ctx[start_0:end_0]
        audio_1 = audio_1_ctx[start_1:end_1]

        return (audio_0, audio_1, torch.tensor(label, dtype=torch.float32))


@torch.no_grad()
def compute_val_loss(
    model: BEATsEncoderAndMLP,
    val_dataloader: torch.utils.data.DataLoader,
    *,
    pos_loss_weight: float,
    max_batches: int,
) -> float:
    """Compute mean BCE-with-logits validation loss over the val dataloader.

    max_batches: <=0 means all batches.
    """

    losses = []
    n = 0
    for audio_0, audio_1, labels in tqdm(val_dataloader):
        n += 1
        if int(max_batches) > 0 and n > int(max_batches):
            break

        audio_0 = audio_0.to(device)
        audio_1 = audio_1.to(device)
        labels = labels.to(device)

        logits = model(audio_0, audio_1)
        pos_weight = torch.tensor(float(pos_loss_weight), device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean", pos_weight=pos_weight)

        losses.append(float(loss.detach().float().cpu().item()))

    if not losses:
        return float("nan")
    return float(np.mean(losses))


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    args = parse_args()
    args.hyperparam = load_hyperparams(args.config)
    hp = args.hyperparam

    require_hyperparams(
        hp,
        [
            "n_epochs",
            "train_batch_size",
            "val_batch_size",
            "val_max_batches",
            "lr",
            "pos_loss_weight",
            "lambda_contrastive",
            "max_error",
            "min_neg_offset",
            "neg_curriculum_epochs",
            "neg_curriculum_min_abs",
            "neg_prob",
            "window_size",
            "time_bins",
            "silence_mode",
            "silence_hist_bins",
            "silence_percentile_fallback",
            "unfreeze_last_n_layers",
            "encoder_lr_mult",
        ],
        config_fp=args.config,
    )

    # Convex mixing weight for auxiliary contrastive loss.
    # (Named lambda_contrastive in YAML; "lambda" itself is a Python keyword.)
    lambda_contrastive = float(hp.lambda_contrastive)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize model
    model = BEATsEncoderAndMLP(args.beats_fp, time_bins=int(hp.time_bins))
    model.to(device)

    # Freeze encoder by default; allow unfreezing last N layers for iteration.
    model.freeze_encoder(unfreeze_last_n_layers=int(hp.unfreeze_last_n_layers))

    # Initialize data
    train_dataset = AudioPairDataset(
        args.train_dir,
        window_size_sec=float(hp.window_size),
        max_error_sec=float(hp.max_error),
        min_neg_offset_sec=float(hp.min_neg_offset),
        neg_prob=float(hp.neg_prob),
    )
    val_dataset = AudioPairDataset(
        args.val_dir,
        window_size_sec=float(hp.window_size),
        max_error_sec=float(hp.max_error),
        min_neg_offset_sec=float(hp.min_neg_offset),
        neg_prob=float(hp.neg_prob),
        deterministic=True,
        deterministic_seed=1337,
    )

    # Silence detection: compute threshold from train RMS histogram and apply to training only.
    silence_threshold_db = None
    if str(hp.silence_mode) != "off":
        train_rms = np.asarray(train_dataset.anchor_rms_db, dtype=float)
        train_rms = train_rms[np.isfinite(train_rms)]
        if train_rms.size:
            fallback = float(np.percentile(train_rms, float(hp.silence_percentile_fallback)))
            thr = otsu_threshold(train_rms, n_bins=int(hp.silence_hist_bins))
            if thr is None:
                thr = fallback
            else:
                # Keep gating lenient: never more aggressive than the fallback percentile.
                thr = min(float(thr), float(fallback))
            silence_threshold_db = float(thr)
            train_dataset.apply_silence_threshold(silence_threshold_db)

            # Keep validation ungated for honest monitoring; report below-threshold fraction.
            val_rms = np.asarray(val_dataset.anchor_rms_db, dtype=float)
            val_rms = val_rms[np.isfinite(val_rms)]
            if val_rms.size:
                frac_below = float(np.mean(val_rms < float(silence_threshold_db)))
                print(
                    "Val silence diagnostics: "
                    f"frac_below_train_threshold={frac_below:.3f} (threshold_db={silence_threshold_db:.2f})"
                )

            meta = {
                "threshold_db": silence_threshold_db,
                "mode": "auto_hist",
                "hist_bins": int(hp.silence_hist_bins),
                "percentile_fallback": float(hp.silence_percentile_fallback),
                "window_size": float(hp.window_size),
                "applied_to": "train_only",
            }
            out_fp = os.path.join(args.output_dir, "silence_threshold.json")
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved silence threshold to {out_fp}: threshold_db={silence_threshold_db:.2f}")
        else:
            print("Warning: no RMS values available for silence detection; skipping gating")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(hp.train_batch_size),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=(int(hp.val_batch_size) if int(hp.val_batch_size) > 0 else int(hp.train_batch_size)),
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Initialize optimizer (separate LR for the unfrozen encoder layers).
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": float(hp.lr)}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(hp.lr) * float(hp.encoder_lr_mult)})
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)

    n_trainable_encoder = int(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
    n_trainable_head = int(sum(p.numel() for p in model.head.parameters() if p.requires_grad))
    print(f"Trainable params: encoder={n_trainable_encoder}, head={n_trainable_head}")

    # Stable class balance regardless of batch composition
    # (pos_weight>1 emphasizes positives; default keeps it balanced)
    pos_loss_weight = float(hp.pos_loss_weight)

    # Train model (select best by validation loss).
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(int(hp.n_epochs)):
        # Train steps
        print(f"Train Epoch {epoch}")

        # Negative curriculum: start with larger-magnitude negatives to ensure learnable signal,
        # then introduce smaller deltas once the loss begins dropping.
        if epoch < int(hp.neg_curriculum_epochs):
            min_abs = max(float(hp.min_neg_offset), float(hp.neg_curriculum_min_abs))
        else:
            min_abs = float(hp.min_neg_offset)
        train_dataset.neg_curriculum_min_abs_sec = float(min_abs)
        # Keep validation distribution stable across epochs.
        val_dataset.neg_curriculum_min_abs_sec = float(hp.min_neg_offset)

        model.train()
        losses = []
        main_losses = []
        aux_losses = []
        for audio_0, audio_1, labels in tqdm(train_dataloader):
            audio_0 = audio_0.to(device)
            audio_1 = audio_1.to(device)
            labels = labels.to(device)

            feats_0 = model.embed(audio_0)
            feats_1 = model.embed(audio_1)
            feats = model.pair_features(feats_0, feats_1)
            logits = model.head(feats).squeeze(-1)

            pos_weight = torch.tensor(float(pos_loss_weight), device=logits.device)
            main_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean", pos_weight=pos_weight)

            # Default: train on the main BCE loss.
            loss = main_loss

            aux_loss = torch.tensor(0.0, device=logits.device)
            if lambda_contrastive > 0.0:
                pos_mask = labels > 0.5
                n_pos = int(pos_mask.sum().detach().cpu().item())
                if n_pos >= 2:
                    z0 = feats_0[pos_mask]
                    z1 = feats_1[pos_mask]
                    b = int(z0.shape[0])
                    z0_bb = z0.unsqueeze(1).expand(b, b, -1).reshape(b * b, -1)
                    z1_bb = z1.unsqueeze(0).expand(b, b, -1).reshape(b * b, -1)
                    feats_bb = model.pair_features(z0_bb, z1_bb)
                    logits_bb = model.head(feats_bb).reshape(b, b)
                    targets = torch.eye(b, device=logits_bb.device, dtype=logits_bb.dtype)
                    pw = torch.tensor(float(b - 1), device=logits_bb.device)
                    aux_loss = F.binary_cross_entropy_with_logits(logits_bb, targets, reduction="mean", pos_weight=pw)

                    # Convex combination: lambda=0 -> pure main loss, lambda=1 -> pure contrastive aux.
                    loss = (1.0 - lambda_contrastive) * main_loss + lambda_contrastive * aux_loss
            losses.append(float(loss.detach().cpu().item()))
            main_losses.append(float(main_loss.detach().cpu().item()))
            aux_losses.append(float(aux_loss.detach().cpu().item()))

            # Backpropagation
            if not torch.isfinite(loss.detach()).item():
                print("Warning: non-finite loss encountered; skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            # grad clipping for stability (especially when unfreezing encoder layers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(
            "Train Loss: "
            f"total={float(np.mean(losses)):.6g} "
            f"main={float(np.mean(main_losses)):.6g} "
            f"aux={float(np.mean(aux_losses)):.6g} "
            f"(lambda_contrastive={lambda_contrastive:.3g})"
        )

        # Val steps
        print(f"Val Epoch {epoch}")
        model.eval()
        with torch.no_grad():
            val_loss = compute_val_loss(
                model,
                val_dataloader,
                pos_loss_weight=float(pos_loss_weight),
                max_batches=int(hp.val_max_batches),
            )
        print(f"Val Loss: {val_loss}")

        if np.isfinite(val_loss) and (val_loss < best_val_loss - 1e-12):
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_fp = os.path.join(args.output_dir, "model_best.pt")
            torch.save(model.state_dict(), best_fp)
            print(f"Saved best model to {best_fp} (epoch={best_epoch}, val_loss={best_val_loss:.6f})")

    print("Training complete")

    # Save weights
    last_fp = os.path.join(args.output_dir, "model_last.pt")
    torch.save(model.state_dict(), last_fp)
    print(f"Saved final model to {last_fp}")

    # Keep a stable default filename. Prefer the best checkpoint if available.
    output_fp = os.path.join(args.output_dir, "model.pt")
    best_fp = os.path.join(args.output_dir, "model_best.pt")
    if os.path.exists(best_fp):
        # Copy semantics without importing shutil; state_dict is small enough to reload.
        state = torch.load(best_fp, map_location="cpu", weights_only=True)
        torch.save(state, output_fp)
        print(f"Saved best model as {output_fp} (epoch={best_epoch}, val_loss={best_val_loss:.6f})")
    else:
        torch.save(model.state_dict(), output_fp)
        print(f"Saved model to {output_fp}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    parser.add_argument("--train-dir", type=str, required=True, help="path to train directory")
    parser.add_argument("--val-dir", type=str, required=True, help="path to val directory")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config containing hyperparameters (loaded into args.hyperparam.*)",
    )
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

