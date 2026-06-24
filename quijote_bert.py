"""
quijote_bert.py — BERT propio desde cero en PyTorch puro
=========================================================
Sin HuggingFace. Implementación completa de BERT con:

  • MLM  (Masked Language Modeling)  — aprende contexto bidireccional
  • NSP  (Next Sentence Prediction)  — aprende coherencia entre oraciones
  • QA   (Question Answering)        — encuentra respuestas en fragmentos
  • Embeddings de oración para RAG   — reemplaza QuijoteEmbedder

Arquitectura:
  • Transformer bidireccional (sin causal mask)
  • Segment embeddings (oración A vs B)
  • [CLS] token para clasificación
  • [MASK] token para MLM
  • [SEP] token para separar oraciones

Config recomendada RPi5:
  d_model=128, n_heads=4, n_layers=4 → ~8M params, ~32MB

Config recomendada laptop:
  d_model=256, n_heads=8, n_layers=6 → ~22M params, ~85MB
"""

import os
import json
import random
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
#  TOKENS ESPECIALES BERT
# ══════════════════════════════════════════════════════════

BERT_SPECIAL = {
    "[PAD]":  0,
    "[UNK]":  1,
    "[CLS]":  2,   # inicio de secuencia — vector de clasificación
    "[SEP]":  3,   # separador entre oraciones
    "[MASK]": 4,   # token enmascarado para MLM
}


# ══════════════════════════════════════════════════════════
#  EMBEDDINGS BERT
# ══════════════════════════════════════════════════════════

class BertEmbeddings(nn.Module):
    """
    BERT usa 3 tipos de embeddings sumados:
      1. Token embedding     — qué token es
      2. Position embedding  — posición en la secuencia
      3. Segment embedding   — ¿es oración A o B? (para NSP)
    """

    def __init__(self, vocab_size: int, d_model: int,
                 max_seq: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(max_seq, d_model)
        self.segment_emb  = nn.Embedding(2, d_model)   # 0=oración A, 1=oración B
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.max_seq = max_seq

        # Posiciones fijas
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq).unsqueeze(0)
        )

    def forward(self, input_ids: torch.Tensor,
                segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = self.position_ids[:, :T]

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        emb = self.token_emb(input_ids) + self.position_emb(pos) + self.segment_emb(segment_ids)
        return self.drop(self.norm(emb))


# ══════════════════════════════════════════════════════════
#  BLOQUE DE ATENCIÓN BERT (bidireccional, sin causal mask)
# ══════════════════════════════════════════════════════════

class BertAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (B, T) → True donde hay padding
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class BertBlock(nn.Module):
    """Bloque BERT estándar: atención + FFN con post-norm (igual al paper original)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = BertAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


# ══════════════════════════════════════════════════════════
#  MODELO BERT COMPLETO
# ══════════════════════════════════════════════════════════

class QuijoteBERT(nn.Module):
    """
    BERT propio desde cero con:
      • MLM  — preentrenamiento por enmascaramiento
      • NSP  — predicción de siguiente oración
      • QA   — extracción de respuestas (start/end span)
      • Embeddings de oración para RAG (vector [CLS])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 256,
        n_heads:    int = 8,
        n_layers:   int = 6,
        max_seq:    int = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.max_seq  = max_seq

        self.embeddings = BertEmbeddings(vocab_size, d_model, max_seq, dropout)
        self.blocks = nn.ModuleList([
            BertBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Cabezas de tarea ──────────────────────────────

        # MLM: predice el token enmascarado
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # NSP: predice si oración B sigue a oración A
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),   # 0=no sigue, 1=sigue
        )

        # QA: predice posición inicio y fin de la respuesta
        self.qa_head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, input_ids: torch.Tensor,
               segment_ids: Optional[torch.Tensor] = None,
               pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encoder principal — retorna representaciones de todos los tokens.
        input_ids:   (B, T)
        segment_ids: (B, T) — 0 para oración A, 1 para oración B
        pad_mask:    (B, T) — True donde hay padding
        Retorna: (B, T, d_model)
        """
        x = self.embeddings(input_ids, segment_ids)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x)

    def forward_mlm(self, input_ids: torch.Tensor,
                    segment_ids: Optional[torch.Tensor] = None,
                    pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retorna logits para MLM: (B, T, vocab_size)"""
        hidden = self.encode(input_ids, segment_ids, pad_mask)
        return self.mlm_head(hidden)

    def forward_nsp(self, input_ids: torch.Tensor,
                    segment_ids: Optional[torch.Tensor] = None,
                    pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retorna logits NSP desde vector [CLS]: (B, 2)"""
        hidden = self.encode(input_ids, segment_ids, pad_mask)
        cls_vec = hidden[:, 0, :]   # primer token = [CLS]
        return self.nsp_head(cls_vec)

    def forward_qa(self, input_ids: torch.Tensor,
                   segment_ids: Optional[torch.Tensor] = None,
                   pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna logits de inicio y fin para QA.
        Retorna: (start_logits, end_logits) cada uno (B, T)
        """
        hidden = self.encode(input_ids, segment_ids, pad_mask)
        logits = self.qa_head(hidden)   # (B, T, 2)
        return logits[:, :, 0], logits[:, :, 1]

    def get_sentence_embedding(self, input_ids: torch.Tensor,
                                segment_ids: Optional[torch.Tensor] = None,
                                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retorna embedding de oración normalizado desde [CLS].
        Usado para RAG — reemplaza QuijoteEmbedder.
        Retorna: (B, d_model)
        """
        hidden = self.encode(input_ids, segment_ids, pad_mask)
        cls_vec = hidden[:, 0, :]
        return F.normalize(cls_vec, dim=-1)

    def encode_texts(self, texts: List[str], tokenizer,
                     batch_size: int = 32,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Encodea una lista de textos a embeddings para RAG.
        Compatible con la interfaz de QuijoteEmbedder.
        """
        self.eval()
        all_embs = []
        mask_id  = tokenizer.token2id.get("[MASK]", tokenizer.unk_id)
        cls_id   = tokenizer.token2id.get("[CLS]",  tokenizer.bos_id)
        sep_id   = tokenizer.token2id.get("[SEP]",  tokenizer.eos_id)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            seqs  = []
            for t in batch:
                ids = [cls_id] + tokenizer.encode(t)[:self.max_seq - 2] + [sep_id]
                seqs.append(ids)

            max_len  = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), tokenizer.pad_id,
                                   dtype=torch.long, device=device)
            pad_mask  = torch.ones((len(seqs), max_len), dtype=torch.bool, device=device)

            for j, seq in enumerate(seqs):
                input_ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                pad_mask[j, :len(seq)]  = False

            with torch.no_grad():
                emb = self.get_sentence_embedding(input_ids, pad_mask=pad_mask)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    def answer_question(self, question: str, context: str,
                        tokenizer, device: torch.device,
                        max_answer_len: int = 50) -> str:
        """
        Encuentra la respuesta a una pregunta dentro de un contexto.
        Formato: [CLS] pregunta [SEP] contexto [SEP]
        """
        self.eval()
        cls_id = tokenizer.token2id.get("[CLS]", tokenizer.bos_id)
        sep_id = tokenizer.token2id.get("[SEP]", tokenizer.eos_id)

        q_ids   = tokenizer.encode(question)
        ctx_ids = tokenizer.encode(context)

        # Truncar para que quepa en max_seq
        max_ctx = self.max_seq - len(q_ids) - 3
        ctx_ids = ctx_ids[:max_ctx]

        # [CLS] pregunta [SEP] contexto [SEP]
        input_ids   = [cls_id] + q_ids + [sep_id] + ctx_ids + [sep_id]
        segment_ids = [0] * (len(q_ids) + 2) + [1] * (len(ctx_ids) + 1)

        # Offset donde empieza el contexto en la secuencia
        ctx_offset = len(q_ids) + 2

        ids_t = torch.tensor([input_ids],   dtype=torch.long, device=device)
        seg_t = torch.tensor([segment_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            start_logits, end_logits = self.forward_qa(ids_t, seg_t)

        # Buscar el mejor span dentro del contexto
        start_logits = start_logits[0, ctx_offset:ctx_offset + len(ctx_ids)]
        end_logits   = end_logits[0,   ctx_offset:ctx_offset + len(ctx_ids)]

        # Encontrar el span de mayor puntuación
        best_score = float("-inf")
        best_start, best_end = 0, 0

        for s in range(len(ctx_ids)):
            for e in range(s, min(s + max_answer_len, len(ctx_ids))):
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

        answer_ids = ctx_ids[best_start:best_end + 1]
        return tokenizer.decode(answer_ids)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        config = dict(
            vocab_size = self.embeddings.token_emb.num_embeddings,
            d_model    = self.d_model,
            n_heads    = self.blocks[0].attn.n_heads,
            n_layers   = self.n_layers,
            max_seq    = self.max_seq,
        )
        with open(os.path.join(path, "bert_config.json"), "w") as f:
            json.dump(config, f)
        torch.save(self.state_dict(), os.path.join(path, "bert.pt"))

    @classmethod
    def load(cls, path: str, device: torch.device) -> "QuijoteBERT":
        with open(os.path.join(path, "bert_config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        model.load_state_dict(
            torch.load(os.path.join(path, "bert.pt"),
                       map_location=device, weights_only=False)
        )
        model.to(device)
        return model


# ══════════════════════════════════════════════════════════
#  PREPARACIÓN DE DATOS BERT
# ══════════════════════════════════════════════════════════

class BertDataPrep:
    """
    Genera los datos de entrenamiento para BERT:
      • MLM:  enmascara 15% de tokens aleatoriamente
      • NSP:  50% oraciones consecutivas, 50% aleatorias
    """

    MASK_PROB     = 0.15   # probabilidad de enmascarar un token
    MASK_TOKEN    = 0.80   # 80% reemplazar con [MASK]
    RANDOM_TOKEN  = 0.10   # 10% reemplazar con token aleatorio
    KEEP_TOKEN    = 0.10   # 10% mantener el token original

    def __init__(self, tokenizer, max_seq: int = 512):
        self.tok     = tokenizer
        self.max_seq = max_seq
        self.mask_id = tokenizer.token2id.get("[MASK]", tokenizer.unk_id)
        self.cls_id  = tokenizer.token2id.get("[CLS]",  tokenizer.bos_id)
        self.sep_id  = tokenizer.token2id.get("[SEP]",  tokenizer.eos_id)
        self.vocab_size = len(tokenizer.token2id)

    def mask_tokens(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Aplica MLM masking.
        Retorna (ids_enmascarados, labels) donde labels=-100 para no enmascarados.
        """
        masked = list(ids)
        labels = [-100] * len(ids)

        for i, token_id in enumerate(ids):
            if random.random() < self.MASK_PROB:
                labels[i] = token_id   # guardar original como label

                r = random.random()
                if r < self.MASK_TOKEN:
                    masked[i] = self.mask_id
                elif r < self.MASK_TOKEN + self.RANDOM_TOKEN:
                    masked[i] = random.randint(5, self.vocab_size - 1)
                # else: mantener el original

        return masked, labels

    def create_nsp_pair(self, sentences: List[List[int]]) -> Tuple[List[int], List[int], int]:
        """
        Crea un par de oraciones para NSP.
        Retorna (input_ids, segment_ids, is_next)
          is_next=1 si B sigue a A, 0 si es aleatoria
        """
        idx = random.randint(0, len(sentences) - 2)
        a   = sentences[idx]

        if random.random() < 0.5:
            b       = sentences[idx + 1]
            is_next = 1
        else:
            b       = sentences[random.randint(0, len(sentences) - 1)]
            is_next = 0

        # Truncar para que quepan: [CLS] A [SEP] B [SEP]
        max_len = self.max_seq - 3
        while len(a) + len(b) > max_len:
            if len(a) > len(b):
                a = a[:-1]
            else:
                b = b[:-1]

        input_ids   = [self.cls_id] + a + [self.sep_id] + b + [self.sep_id]
        segment_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)

        return input_ids, segment_ids, is_next

    def prepare_batch(self, sentences: List[List[int]], batch_size: int = 8):
        """Genera un batch completo de datos MLM + NSP."""
        batch_input    = []
        batch_segments = []
        batch_labels   = []
        batch_nsp      = []
        batch_masks    = []

        for _ in range(batch_size):
            if len(sentences) < 2:
                break

            input_ids, segment_ids, is_next = self.create_nsp_pair(sentences)
            masked_ids, labels = self.mask_tokens(input_ids)

            batch_input.append(masked_ids)
            batch_segments.append(segment_ids)
            batch_labels.append(labels)
            batch_nsp.append(is_next)

        if not batch_input:
            return None

        # Padding
        max_len = max(len(x) for x in batch_input)
        def pad(seq, val, length):
            return seq + [val] * (length - len(seq))

        input_t    = torch.tensor([pad(x, 0, max_len)    for x in batch_input],    dtype=torch.long)
        segment_t  = torch.tensor([pad(x, 0, max_len)    for x in batch_segments], dtype=torch.long)
        labels_t   = torch.tensor([pad(x, -100, max_len) for x in batch_labels],   dtype=torch.long)
        nsp_t      = torch.tensor(batch_nsp, dtype=torch.long)
        pad_mask_t = (input_t == 0)

        return input_t, segment_t, labels_t, nsp_t, pad_mask_t
