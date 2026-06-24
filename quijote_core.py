"""
quijote_core.py — Motor Transformer 100% PyTorch puro
======================================================
Sin HuggingFace, sin sentence-transformers, sin FAISS.

Contiene:
  • BPETokenizer            — tokenizador Byte-Pair Encoding propio
  • MultiHeadAttention      — atención multi-cabeza con causal mask
  • MultiHeadAttentionTurbo — atención con KV cache TurboQuant (inferencia)
  • TransformerBlock        — bloque con RMSNorm pre-norm
  • QuijoteGPT              — modelo generativo causal con TurboQuant
  • QuijoteEmbedder         — encoder bidireccional ligero para RAG
  • VectorIndex             — índice vectorial coseno propio (sin FAISS)
"""

import math
import json
import re
import os
import pickle
import struct
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
#  1. TOKENIZADOR BPE PROPIO
# ══════════════════════════════════════════════════════════

class BPETokenizer:
    """
    Byte-Pair Encoding minimalista entrenable desde cero.
    Empieza desde caracteres individuales y fusiona los pares
    más frecuentes iterativamente.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL = [PAD, UNK, BOS, EOS]

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

    # ── Entrenamiento BPE ─────────────────────────────────

    def train(self, texts: List[str], verbose: bool = True):
        # Vocabulario inicial: caracteres únicos + especiales
        chars = set("".join(texts))
        vocab = list(self.SPECIAL) + sorted(chars)

        # Representar cada palabra como secuencia de caracteres + </w>
        word_freq: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freq[" ".join(list(word)) + " </w>"] += 1

        # Convertir a lista mutable
        vocab_words = {w: freq for w, freq in word_freq.items()}

        merges = []
        current_vocab = set(vocab)

        num_merges = self.vocab_size - len(vocab)
        for i in range(num_merges):
            pairs = self._get_pairs(vocab_words)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)

            # Fusionar el par en todas las palabras
            vocab_words = self._merge_pair(best, vocab_words)
            new_token = "".join(best)
            current_vocab.add(new_token)

            if verbose and (i + 1) % 500 == 0:
                print(f"  BPE merge {i+1}/{num_merges} — vocab: {len(current_vocab)}")

        self.merges = merges
        self._build_vocab(current_vocab)
        self._trained = True
        print(f"✅ BPE entrenado: {len(self.token2id)} tokens")

    def _get_pairs(self, vocab_words: dict) -> Counter:
        pairs: Counter = Counter()
        for word, freq in vocab_words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], vocab_words: dict) -> dict:
        result = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        replacement = "".join(pair)
        for word, freq in vocab_words.items():
            new_word = pattern.sub(replacement, word)
            result[new_word] = freq
        return result

    def _build_vocab(self, tokens: set):
        all_tokens = list(self.SPECIAL) + sorted(tokens - set(self.SPECIAL))
        self.token2id = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    # ── Encode / Decode ───────────────────────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        symbols = list(word) + ["</w>"]
        for a, b in self.merges:
            i = 0
            new_syms = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.token2id[self.BOS])
        for word in text.lower().split():
            for sym in self._tokenize_word(word):
                tokens.append(self.token2id.get(sym, self.token2id[self.UNK]))
        if add_eos:
            tokens.append(self.token2id[self.EOS])
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        # Quitar tokens especiales
        for s in self.SPECIAL:
            text = text.replace(s, "")
        return text.strip()

    @property
    def pad_id(self):  return self.token2id[self.PAD]
    @property
    def eos_id(self):  return self.token2id[self.EOS]
    @property
    def bos_id(self):  return self.token2id[self.BOS]
    @property
    def unk_id(self):  return self.token2id[self.UNK]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        data = {"vocab_size": self.vocab_size, "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()},
                "merges": self.merges}
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k, v in data["id2token"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._trained = True
        return tok


# ══════════════════════════════════════════════════════════
#  2. BLOQUES TRANSFORMER
# ══════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """RoPE — Rotary Position Embedding (sin parámetros entrenables)."""

    def __init__(self, dim: int, max_seq: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, d)

        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """SwiGLU — más eficiente que ReLU/GELU clásico."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * expansion * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block (más estable que post-norm en modelos pequeños)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 causal: bool = True, use_turbo: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        if use_turbo and causal:
            from turbo_quant import MultiHeadAttentionTurbo
            self.attn = MultiHeadAttentionTurbo(d_model, n_heads, dropout, causal)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════
#  3. MODELO GENERATIVO (QuijoteGPT)
# ══════════════════════════════════════════════════════════

class QuijoteGPT(nn.Module):
    """
    Modelo causal estilo GPT con soporte TurboQuant.
    Config recomendada para RPi 5 8GB:
      d_model=256, n_heads=4, n_layers=4  → ~15M params, ~60 MB
    En inferencia activa KV cache TurboQuant → ~87% menos RAM en cache.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq: int = 512,
        dropout: float = 0.1,
        use_turbo: bool = False,  # desactivado — tiene conflictos con CUDA
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.use_turbo = use_turbo

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, causal=True, use_turbo=use_turbo)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding y head comparten pesos
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _enable_turbo_cache(self):
        """Activa KV cache TurboQuant en todos los bloques."""
        for block in self.blocks:
            if hasattr(block.attn, "enable_turbo_cache"):
                block.attn.enable_turbo_cache(next(self.parameters()).device)

    def _disable_turbo_cache(self):
        """Desactiva y limpia la KV cache."""
        for block in self.blocks:
            if hasattr(block.attn, "disable_turbo_cache"):
                block.attn.disable_turbo_cache()

    def turbo_memory_info(self) -> str:
        """Muestra info del ahorro de RAM con TurboQuant."""
        for block in self.blocks:
            if hasattr(block.attn, "_cache") and block.attn._cache:
                ratio = block.attn._cache.memory_saved_ratio()
                tokens = block.attn._cache.size()
                return f"TurboQuant activo | {tokens} tokens | ahorro RAM: {ratio*100:.0f}%"
        return "TurboQuant inactivo"

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_id: int = 3,
        device: torch.device = torch.device("cpu"),
    ) -> List[int]:
        self.eval()

        # Activar KV cache TurboQuant para ahorrar RAM
        if self.use_turbo:
            self._enable_turbo_cache()

        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = list(prompt_ids)
        past_ids: Counter = Counter(prompt_ids)

        # Procesar el prompt completo de una vez (llena la cache)
        if self.use_turbo and len(prompt_ids) > 1:
            # Pasar el prompt token a token para llenar la cache
            for i in range(len(prompt_ids)):
                tok = torch.tensor([[prompt_ids[i]]], dtype=torch.long, device=device)
                _ = self(tok)

        for _ in range(max_new_tokens):
            if self.use_turbo:
                # Solo el último token — la cache tiene el resto
                context = ids[:, -1:]
            else:
                context = ids[:, -self.max_seq:]

            logits = self(context)[:, -1, :]   # (1, vocab)

            # Penalización por repetición
            for tid, cnt in past_ids.items():
                logits[0, tid] /= (repetition_penalty ** cnt)

            logits = logits / max(temperature, 1e-6)

            # Top-K
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")

            # Top-P (nucleus)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove = cum_probs - sorted_probs > top_p
            sorted_probs[remove] = 0.0
            sorted_probs /= sorted_probs.sum()

            next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].item()

            generated.append(next_token)
            past_ids[next_token] += 1
            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

            if next_token == eos_id:
                break

        # Desactivar cache al terminar para liberar RAM
        if self.use_turbo:
            self._disable_turbo_cache()

        return generated[len(prompt_ids):]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        config = dict(
            vocab_size=self.embed.num_embeddings,
            d_model=self.d_model,
            n_heads=self.blocks[0].attn.n_heads,
            n_layers=len(self.blocks),
            max_seq=self.max_seq,
            use_turbo=self.use_turbo,
        )
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def load(cls, path: str, device: torch.device) -> "QuijoteGPT":
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        state = torch.load(os.path.join(path, "model.pt"), map_location=device, weights_only=False)
        # Compatibilidad: renombrar _rope -> rope si fue guardado con version anterior
        fixed = {k.replace("attn._rope.", "attn.rope."): v for k, v in state.items()}
        model.load_state_dict(fixed, strict=False)
        model.to(device)
        return model


# ══════════════════════════════════════════════════════════
#  4. EMBEDDER PROPIO (QuijoteEmbedder)
# ══════════════════════════════════════════════════════════

class QuijoteEmbedder(nn.Module):
    """
    Encoder bidireccional ligero para RAG.
    Usa pooling promedio sobre los tokens para obtener
    un vector de oración de dimensión d_model.
    Config recomendada RPi: d_model=128, n_heads=4, n_layers=2 → ~4M params
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, causal=False)  # bidireccional
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retorna embedding de oración (B, d_model)."""
        emb = self.drop(self.embed(x))
        for block in self.blocks:
            emb = block(emb, pad_mask)
        emb = self.norm(emb)

        # Mean pooling ignorando padding
        if pad_mask is not None:
            lengths = (~pad_mask).sum(dim=1, keepdim=True).float().unsqueeze(-1)
            emb = emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            emb = emb.sum(dim=1) / lengths.squeeze(-1).clamp(min=1)
        else:
            emb = emb.mean(dim=1)

        return F.normalize(emb, dim=-1)  # normalizado → similitud coseno directa

    def encode(self, texts: List[str], tokenizer: "BPETokenizer",
               batch_size: int = 32, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        self.eval()
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            seqs = [tokenizer.encode(t) for t in batch_texts]
            max_len = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            pad_mask = torch.ones((len(seqs), max_len), dtype=torch.bool, device=device)
            for j, seq in enumerate(seqs):
                ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                pad_mask[j, :len(seq)] = False

            with torch.no_grad():
                emb = self(ids, pad_mask)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        config = dict(
            vocab_size=self.embed.num_embeddings,
            d_model=self.d_model,
            n_heads=self.blocks[0].attn.n_heads,
            n_layers=len(self.blocks),
        )
        with open(os.path.join(path, "embedder_config.json"), "w") as f:
            json.dump(config, f)
        torch.save(self.state_dict(), os.path.join(path, "embedder.pt"))

    @classmethod
    def load(cls, path: str, device: torch.device) -> "QuijoteEmbedder":
        with open(os.path.join(path, "embedder_config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        model.load_state_dict(torch.load(os.path.join(path, "embedder.pt"), map_location=device, weights_only=False))
        model.to(device)
        return model


# ══════════════════════════════════════════════════════════
#  5. ÍNDICE VECTORIAL PROPIO (sin FAISS)
# ══════════════════════════════════════════════════════════

class VectorIndex:
    """
    Índice de similitud coseno en PyTorch puro.
    Para colecciones pequeñas (<50k docs) es suficientemente rápido en CPU.
    Los vectores se normalizan al insertar → similitud = dot product.
    """

    def __init__(self):
        self.vectors: Optional[torch.Tensor] = None  # (N, d)
        self.documents: List[str] = []

    def build(self, embeddings: torch.Tensor, documents: List[str]):
        assert embeddings.shape[0] == len(documents)
        self.vectors = F.normalize(embeddings.float(), dim=-1)
        self.documents = documents

    def search(self, query_emb: torch.Tensor, top_k: int = 3) -> List[Tuple[float, str]]:
        if self.vectors is None:
            raise RuntimeError("Índice vacío. Llama a build() primero.")
        q = F.normalize(query_emb.float().unsqueeze(0), dim=-1)  # (1, d)
        scores = (self.vectors @ q.T).squeeze(-1)                 # (N,)
        top_scores, top_idx = torch.topk(scores, min(top_k, len(self.documents)))
        return [(top_scores[i].item(), self.documents[top_idx[i].item()])
                for i in range(len(top_idx))]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.vectors, os.path.join(path, "vectors.pt"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> bool:
        vpath = os.path.join(path, "vectors.pt")
        dpath = os.path.join(path, "documents.pkl")
        if os.path.exists(vpath) and os.path.exists(dpath):
            self.vectors = torch.load(vpath, map_location="cpu", weights_only=False)
            with open(dpath, "rb") as f:
                self.documents = pickle.load(f)
            return True
        return False