import torch
from torch import nn
import torchaudio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class BEATsEncoderAndMLP(nn.Module):
    def __init__(self, beats_checkpoint_fp, time_bins: int = 1):
        super().__init__()
        from beats import BEATs, BEATsConfig
        try:
            # PyTorch 2.6+ defaults to weights_only=True; BEATs checkpoints include non-tensor
            # metadata (e.g., cfg), so we explicitly opt out.
            beats_ckpt = torch.load(beats_checkpoint_fp, map_location="cpu", weights_only=False)
        except EOFError as e:
            size = None
            try:
                size = os.path.getsize(beats_checkpoint_fp)
            except OSError:
                pass
            size_msg = f" (size={size} bytes)" if size is not None else ""
            raise RuntimeError(
                f"Failed to load BEATs checkpoint at '{beats_checkpoint_fp}'{size_msg}. "
                "This usually means the file is truncated/corrupt (or is an HTML download page)."
            ) from e
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        self.encoder = BEATs(beats_cfg)
        self.encoder.load_state_dict(beats_ckpt['model'])
        embedding_dim = self.encoder.cfg.encoder_embed_dim
        if time_bins < 1:
            raise ValueError("time_bins must be >= 1")
        self.time_bins = int(time_bins)
        self.embedding_dim = embedding_dim

        pooled_dim = embedding_dim * self.time_bins
        self.head = nn.Sequential(
            nn.Linear(3 * pooled_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        
        self.encoder.to(device)
        self.head.to(device)
        
    def embed(self, audio):
        """
        Encode mono audio with BEATs audio encoder
        """
        audio = audio.to(device)
        feats = self.encoder.extract_features(audio, feature_only=True)[0] # [B T C]

        # Temporal pooling:
        # - time_bins=1 matches previous behavior (mean over time)
        # - time_bins>1 preserves coarse temporal structure by mean-pooling within bins
        if self.time_bins == 1:
            return torch.mean(feats, dim=1)

        batch_size, n_frames, n_channels = feats.shape
        if n_frames == 0:
            return torch.zeros((batch_size, n_channels * self.time_bins), device=feats.device, dtype=feats.dtype)

        pooled = []
        for bin_idx in range(self.time_bins):
            start = int(round(bin_idx * n_frames / self.time_bins))
            end = int(round((bin_idx + 1) * n_frames / self.time_bins))
            if end <= start:
                # Extremely short inputs: fall back to the nearest frame
                frame_idx = min(start, n_frames - 1)
                pooled.append(feats[:, frame_idx : frame_idx + 1, :].mean(dim=1))
            else:
                pooled.append(feats[:, start:end, :].mean(dim=1))

        return torch.cat(pooled, dim=-1)

    def pair_features(self, feats_0: torch.Tensor, feats_1: torch.Tensor) -> torch.Tensor:
        """Construct symmetric pair features from two BEATs embeddings.

        Baseline feature: [z0, z1, |z0 - z1|].
        """
        return torch.cat([feats_0, feats_1, torch.abs(feats_0 - feats_1)], dim=-1)
        
    def forward(self, audio_0, audio_1):
        """
        Compute similarity of two mono audio tensors
        """
        feats_0 = self.embed(audio_0)
        feats_1 = self.embed(audio_1)

        feats = self.pair_features(feats_0, feats_1)
        similarity = self.head(feats).squeeze(-1)
        return similarity
    
    def freeze_encoder(self, unfreeze_last_n_layers: int = 0):
        """Freeze the BEATs encoder, optionally unfreezing the last N transformer layers."""
        n = int(unfreeze_last_n_layers)
        if n < 0:
            raise ValueError("unfreeze_last_n_layers must be >= 0")

        for param in self.encoder.parameters():
            param.requires_grad = False

        # By default we keep BEATs in eval mode for stability.
        self.encoder.eval()

        if n == 0:
            return

        # Unfreeze last-N transformer blocks if present.
        layers = None
        try:
            layers = self.encoder.encoder.layers
        except AttributeError:
            layers = None

        if layers is None:
            raise RuntimeError("Could not locate BEATs transformer layers at self.encoder.encoder.layers")

        n = min(n, len(layers))
        for layer in list(layers)[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
