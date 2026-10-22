"""UnCRtainTS (Ebel et al., CVPRW 2023), built from the official repo with the README's
SEN12MS-CR-TS settings, wrapped to the CATFormer interface: (s2, s1, cloud, tpos) -> (mean, logvar, None)."""
import os, sys

import torch
import torch.nn as nn

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uncrtaints_repo", "model")


class UnCRtainTSWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, REPO)
        from parse_args import create_parser
        from src import utils
        from src.backbones import uncrtaints
        cfg = create_parser(mode="train").parse_args([
            "--model", "uncrtaints", "--input_t", "3", "--scale_by", "10.0", "--loss", "MGNLL",
            "--covmode", "diag", "--var_nonLinearity", "softplus", "--use_sar",
            "--block_type", "mbconv", "--n_head", "16"])
        cfg = utils.str2list(cfg, list_args=["encoder_widths", "decoder_widths", "out_conv"])
        cfg.out_conv[-1] += 13  # diag covariance: one variance per band (as train_reconstruct.py does)
        self.net = uncrtaints.UNCRTAINTS(
            input_dim=2 + 13, encoder_widths=cfg.encoder_widths, decoder_widths=cfg.decoder_widths,
            out_conv=cfg.out_conv, out_nonlin_mean=cfg.mean_nonLinearity,
            out_nonlin_var=cfg.var_nonLinearity, agg_mode=cfg.agg_mode,
            encoder_norm=cfg.encoder_norm, decoder_norm=cfg.decoder_norm, n_head=cfg.n_head,
            d_model=cfg.d_model, d_k=cfg.d_k, pad_value=cfg.pad_value,
            padding_mode=cfg.padding_mode, positional_encoding=cfg.positional_encoding,
            covmode=cfg.covmode, scale_by=cfg.scale_by, separate_out=cfg.separate_out,
            use_v=cfg.use_v, block_type=cfg.block_type, is_mono=False)

    def forward(self, s2, s1, cloud, tpos):
        # official BaseModel: inputs * scale_by, mean / scale_by, variance / scale_by**2
        sc = self.net.scale_by
        x = torch.cat([s1, s2], 2) * sc  # official order: S1 then S2 channels
        out = self.net(x, batch_positions=tpos * 30.0)[:, 0]  # dates in days
        mean, var = out[:, :13] / sc, out[:, 13:26] / sc ** 2
        return mean, torch.log(var.clamp_min(1e-8)), None
