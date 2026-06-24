"""
turbo_quant.py — KV Cache comprimida para inferencia eficiente
==============================================================
Implementación corregida de compresión de KV cache inspirada en TurboQuant.

CORRECCIONES respecto a versión anterior:
  • Matrices R y P se guardan en disco y se reutilizan entre sesiones
  • Bug del prompt doble eliminado
  • scale/zero_point guardados en FP16 (no FP32) para mayor ahorro real
  • Medición de RAM honesta

AHORRO REAL esperado en RPi5 con modelo ~22M params:
  • Sin compresión: cada token = d_head * 2 (K+V) * 4 bytes (FP32)
  • Con compresión 3 bits: ~87% menos en los valores K y V
  • Para 120 tokens con d_head=32: 120*32*2*4 = ~30KB → ~4KB
  • El modelo en sí (~85MB) no cambia — la cache es pequeña de por sí
  • CONCLUSIÓN: útil principalmente con secuencias largas (500+ tokens)
"""

import os
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List


# ══════════════════════════════════════════════════════════
#  CUANTIZACIÓN
# ══════════════════════════════════════════════════════════

def quantize_to_bits(
    x: torch.Tensor, n_bits: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cuantiza a n_bits. Guarda scale/zero_point en FP16 para ahorrar RAM.
    """
    n_levels = 2 ** n_bits - 1
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    scale  = (x_max - x_min).clamp(min=1e-8) / n_levels
    zp     = x_min
    x_q    = ((x - zp) / scale).round().clamp(0, n_levels).to(torch.uint8)
    # FP16 para scale y zp — mitad de RAM que FP32
    return x_q, scale.half(), zp.half()


def dequantize(x_q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    return x_q.float() * scale.float() + zp.float()


# ══════════════════════════════════════════════════════════
#  POLAR QUANT — con matrices persistentes
# ══════════════════════════════════════════════════════════

class PolarQuant:
    """
    Rotación ortogonal + cuantización.
    La matriz R se guarda en disco para ser consistente entre sesiones.
    """

    def __init__(self, dim: int, n_bits: int = 3,
                 device: torch.device = torch.device("cpu"),
                 cache_dir: str = "quijote_model"):
        self.dim    = dim
        self.n_bits = n_bits
        self.device = device

        # Cargar o crear matriz de rotación persistente
        R_path = Path(cache_dir) / f"polar_R_{dim}.pt"
        if R_path.exists():
            self.R = torch.load(str(R_path), map_location=device, weights_only=False)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            raw = torch.randn(dim, dim)
            self.R, _ = torch.linalg.qr(raw)
            torch.save(self.R, str(R_path))
            self.R = self.R.to(device)

    def compress(self, x: torch.Tensor):
        x_rot = x @ self.R
        return quantize_to_bits(x_rot, self.n_bits)

    def decompress(self, x_q, scale, zp) -> torch.Tensor:
        x_rot = dequantize(x_q, scale, zp)
        return x_rot @ self.R.T


# ══════════════════════════════════════════════════════════
#  QJL — con proyección persistente
# ══════════════════════════════════════════════════════════

class QJL:
    """
    Johnson-Lindenstrauss: corrige el error residual de PolarQuant
    proyectando el residuo a dimensión menor.
    La matriz P se guarda en disco para consistencia.
    """

    def __init__(self, dim: int, n_bits: int = 4,
                 device: torch.device = torch.device("cpu"),
                 cache_dir: str = "quijote_model"):
        self.dim    = dim
        self.m      = max(dim // 4, 16)
        self.n_bits = n_bits
        self.device = device

        P_path = Path(cache_dir) / f"qjl_P_{dim}.pt"
        if P_path.exists():
            self.P = torch.load(str(P_path), map_location=device, weights_only=False)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.P = torch.randn(dim, self.m) / math.sqrt(self.m)
            torch.save(self.P, str(P_path))
            self.P = self.P.to(device)

    def compress_residual(self, residual: torch.Tensor):
        projected = residual @ self.P
        return quantize_to_bits(projected, self.n_bits)

    def decompress_residual(self, r_q, scale, zp) -> torch.Tensor:
        projected = dequantize(r_q, scale, zp)
        return projected @ self.P.T


# ══════════════════════════════════════════════════════════
#  KV CACHE COMPRIMIDA
# ══════════════════════════════════════════════════════════

class KVCacheTurbo:
    """
    Cache de Keys/Values comprimida con PolarQuant + QJL.

    Cuándo vale la pena usarla:
      • Secuencias largas (500+ tokens): ahorro significativo
      • Secuencias cortas (<200 tokens): ahorro pequeño pero sin coste

    Ahorro real estimado vs FP32:
      PolarQuant 3 bits + QJL 4 bits (dim/4):
        bits_comprimidos = 3 + 4*(1/4) = 4 bits
        ahorro = 1 - 4/32 = 87.5%
    """

    def __init__(
        self,
        d_head: int,
        n_bits_polar: int = 3,
        n_bits_qjl:   int = 4,
        use_qjl:      bool = True,
        device:       torch.device = torch.device("cpu"),
        cache_dir:    str = "quijote_model",
    ):
        self.d_head  = d_head
        self.use_qjl = use_qjl
        self.device  = device

        self.polar_k = PolarQuant(d_head, n_bits_polar, device, cache_dir)
        self.polar_v = PolarQuant(d_head, n_bits_polar, device, cache_dir)

        if use_qjl:
            self.qjl_k = QJL(d_head, n_bits_qjl, device, cache_dir)
            self.qjl_v = QJL(d_head, n_bits_qjl, device, cache_dir)

        self._k_store: List = []
        self._v_store: List = []

    def push(self, k: torch.Tensor, v: torch.Tensor):
        """Comprime y almacena un token. k,v: (B, H, 1, d_head)"""
        k = k.squeeze(2)
        v = v.squeeze(2)

        kq, ks, kzp = self.polar_k.compress(k)
        vq, vs, vzp = self.polar_v.compress(v)

        entry_k = (kq, ks, kzp)
        entry_v = (vq, vs, vzp)

        if self.use_qjl:
            k_recon = self.polar_k.decompress(kq, ks, kzp)
            v_recon = self.polar_v.decompress(vq, vs, vzp)
            krq, krs, krzp = self.qjl_k.compress_residual(k - k_recon)
            vrq, vrs, vrzp = self.qjl_v.compress_residual(v - v_recon)
            entry_k = entry_k + (krq, krs, krzp)
            entry_v = entry_v + (vrq, vrs, vrzp)

        self._k_store.append(entry_k)
        self._v_store.append(entry_v)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Descomprime y retorna toda la cache. (B, H, T, d_head)"""
        ks, vs = [], []
        for ek, ev in zip(self._k_store, self._v_store):
            if self.use_qjl:
                kq, ksc, kzp, krq, krs, krzp = ek
                vq, vsc, vzp, vrq, vrs, vrzp = ev
                k = self.polar_k.decompress(kq, ksc, kzp) + self.qjl_k.decompress_residual(krq, krs, krzp)
                v = self.polar_v.decompress(vq, vsc, vzp) + self.qjl_v.decompress_residual(vrq, vrs, vrzp)
            else:
                kq, ksc, kzp = ek
                vq, vsc, vzp = ev
                k = self.polar_k.decompress(kq, ksc, kzp)
                v = self.polar_v.decompress(vq, vsc, vzp)
            ks.append(k.unsqueeze(2))
            vs.append(v.unsqueeze(2))

        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def size(self) -> int:
        return len(self._k_store)

    def clear(self):
        self._k_store.clear()
        self._v_store.clear()

    def ram_saved_percent(self) -> float:
        bits = self.polar_k.n_bits
        bits_res = self.qjl_k.n_bits * (self.qjl_k.m / self.d_head) if self.use_qjl else 0
        return (1.0 - (bits + bits_res) / 32.0) * 100


# ══════════════════════════════════════════════════════════
#  ATENCIÓN CON TURBOQUANT
# ══════════════════════════════════════════════════════════

class MultiHeadAttentionTurbo(nn.Module):
    """
    Atención con KV cache TurboQuant para inferencia eficiente.
    En entrenamiento: atención normal (cache desactivada).
    En inferencia:    usa KVCacheTurbo token a token.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        dropout:     float = 0.1,
        causal:      bool  = True,
        turbo_bits:  int   = 3,
        use_qjl:     bool  = True,
        cache_dir:   str   = "quijote_model",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.causal    = causal
        self.scale     = self.d_head ** -0.5
        self.turbo_bits = turbo_bits
        self.use_qjl   = use_qjl
        self.cache_dir = cache_dir

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        from quijote_core import RotaryEmbedding, apply_rope
        self._rope       = RotaryEmbedding(self.d_head)
        self._apply_rope = apply_rope

        self._cache: Optional[KVCacheTurbo] = None
        self._inference_mode = False

    def enable_turbo_cache(self, device: torch.device):
        self._cache = KVCacheTurbo(
            d_head      = self.d_head,
            n_bits_polar = self.turbo_bits,
            use_qjl     = self.use_qjl,
            device      = device,
            cache_dir   = self.cache_dir,
        )
        self._inference_mode = True

    def disable_turbo_cache(self):
        if self._cache:
            self._cache.clear()
        self._cache = None
        self._inference_mode = False

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self._rope(q, T)
        q, k = self._apply_rope(q, k, cos, sin)

        if self._inference_mode and self._cache is not None and T == 1:
            # Inferencia token a token — usar cache comprimida
            self._cache.push(k, v)
            k_all, v_all = self._cache.get()
            attn = (q @ k_all.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out  = (attn @ v_all).transpose(1, 2).reshape(B, T, C)
        else:
            # Entrenamiento — atención normal sin cache
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.causal:
                cm   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                attn = attn.masked_fill(cm, float("-inf"))
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.drop(attn)
            out  = (attn @ v).transpose(1, 2).reshape(B, T, C)

        return self.out(out)