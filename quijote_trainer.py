"""
quijote_trainer.py — Entrenamiento GPT + BERT + RAG completo
============================================================
Sin HuggingFace. Solo PyTorch + quijote_core + quijote_bert.

Cambios v4:
  • QuijoteBERT integrado — reemplaza QuijoteEmbedder en RAG
  • Entrenamiento BERT con MLM + NSP
  • RAG usa BERT para embeddings y QA para respuestas exactas
  • GPT usa BERT como contexto enriquecido
"""

import gc
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from quijote_core import (
    BPETokenizer,
    QuijoteGPT,
    QuijoteEmbedder,
    VectorIndex,
)
from quijote_bert import QuijoteBERT, BertDataPrep

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════
#  TurboQuantOptimizer — LR estable
# ══════════════════════════════════════════════════════════

class TurboQuantOptimizer:
    def __init__(self, base_lr: float = 3e-3, warmup_steps: int = 100):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self._step = 0
        self.current_lr = 0.0

    def step(self, loss_value: float = 0.0) -> float:
        self._step += 1
        if self._step <= self.warmup_steps:
            lr = self.base_lr * (self._step / self.warmup_steps)
        else:
            lr = self.base_lr
        self.current_lr = lr
        return lr


# ══════════════════════════════════════════════════════════
#  Scheduler coseno con warm-up
# ══════════════════════════════════════════════════════════

def make_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-2)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════
#  Dataset GPT
# ══════════════════════════════════════════════════════════

class LMDataset(Dataset):
    def __init__(self, tokenizer: BPETokenizer, texts: List[str], max_length: int = 128):
        self.pad_id = tokenizer.pad_id
        self.examples: List[torch.Tensor] = []
        for text in texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) > 1:
                ids = ids[:max_length]
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_lm(batch: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(t.size(0) for t in batch)
    src = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(batch):
        src[i, :t.size(0)] = t
    return src[:, :-1].contiguous(), src[:, 1:].contiguous()


# ══════════════════════════════════════════════════════════
#  Sistema RAG con BERT + GPT
# ══════════════════════════════════════════════════════════

class QuijoteRAG:
    """
    Sistema RAG completo:
      • BPETokenizer  — vocabulario compartido entre GPT y BERT
      • QuijoteGPT    — genera respuestas en lenguaje natural
      • QuijoteBERT   — entiende preguntas, hace embeddings y QA
      • VectorIndex   — búsqueda por similitud coseno

    Configs disponibles:

    NANO    ~8M  params  ~32MB   RPi con pocos datos
      GPT: d_model=128, n_heads=4, n_layers=3
      BERT: d_model=128, n_heads=4, n_layers=4

    PEQUEÑO ~22M params  ~85MB   RPi5 estándar
      GPT: d_model=256, n_heads=8, n_layers=6
      BERT: d_model=128, n_heads=4, n_layers=6

    MEDIANO ~60M params  ~230MB  RPi5 8GB
      GPT: d_model=512, n_heads=8, n_layers=8
      BERT: d_model=256, n_heads=8, n_layers=6

    GRANDE  ~130M params ~500MB  Laptop  ← actual
      GPT: d_model=768, n_heads=12, n_layers=12
      BERT: d_model=256, n_heads=8, n_layers=8
    """

    GPT_CONFIG  = dict(d_model=256, n_heads=8,  n_layers=6,  max_seq=256, dropout=0.1)
    BERT_CONFIG = dict(d_model=128, n_heads=4,  n_layers=6,  max_seq=256, dropout=0.1)

    def __init__(
        self,
        save_dir:       str = "quijote_model",
        bpe_vocab_size: int = 12000,
        gpt_config:     Optional[dict] = None,
        bert_config:    Optional[dict] = None,
        use_bert:       bool = True,
    ):
        self.save_dir       = Path(save_dir)
        self.device         = _get_device()
        self.bpe_vocab_size = bpe_vocab_size
        self.gpt_cfg        = gpt_config  or self.GPT_CONFIG
        self.bert_cfg       = bert_config or self.BERT_CONFIG
        self.use_bert       = use_bert

        self.tokenizer: Optional[BPETokenizer]  = None
        self.gpt:       Optional[QuijoteGPT]    = None
        self.bert:      Optional[QuijoteBERT]   = None
        self.embedder:  Optional[QuijoteEmbedder] = None  # fallback si no hay BERT
        self.index = VectorIndex()

        log.info(f"🖥️  QuijoteRAG inicializado | device={self.device} | BERT={'✅' if use_bert else '❌'}")

    # ── Entrenamiento completo ────────────────────────────

    def train(
        self,
        documents:  List[str],
        epochs_gpt: int   = 5,
        epochs_bert: int  = 3,
        batch_size: int   = 4,
        max_length: int   = 128,
        grad_accum: int   = 4,
        base_lr:    float = 1e-3,
    ):
        log.info("═" * 55)
        log.info("PASO 1/3 — Entrenando BPE Tokenizer...")
        log.info("═" * 55)
        self.tokenizer = BPETokenizer(vocab_size=self.bpe_vocab_size)
        # Agregar tokens especiales BERT al vocabulario
        self.tokenizer.train(documents, verbose=True)
        self._add_bert_tokens()
        vocab_size = len(self.tokenizer.token2id)

        log.info("═" * 55)
        log.info(f"PASO 2/3 — Entrenando QuijoteGPT ({vocab_size} tokens)...")
        log.info("═" * 55)
        self.gpt = QuijoteGPT(vocab_size=vocab_size, **self.gpt_cfg).to(self.device)
        params = self.gpt.count_params()
        log.info(f"  Parámetros GPT: {params:,} (~{params*4//1_000_000} MB en FP32)")
        self._train_gpt(documents, epochs_gpt, batch_size, max_length, grad_accum, base_lr)

        if self.use_bert:
            log.info("═" * 55)
            log.info(f"PASO 3/3 — Entrenando QuijoteBERT (MLM + NSP)...")
            log.info("═" * 55)
            self.bert = QuijoteBERT(vocab_size=vocab_size, **self.bert_cfg).to(self.device)
            params_bert = self.bert.count_params()
            log.info(f"  Parámetros BERT: {params_bert:,} (~{params_bert*4//1_000_000} MB en FP32)")
            self._train_bert(documents, epochs_bert, batch_size, grad_accum, base_lr)
        else:
            log.info("PASO 3/3 — Entrenando QuijoteEmbedder (fallback)...")
            from quijote_core import QuijoteEmbedder
            emb_cfg = dict(d_model=128, n_heads=4, n_layers=2, max_seq=128, dropout=0.1)
            self.embedder = QuijoteEmbedder(vocab_size=vocab_size, **emb_cfg).to(self.device)
            self._train_embedder(documents, epochs_bert, batch_size, max_length, grad_accum, base_lr)

        log.info("✅ Entrenamiento completo")
        _free()

    def _add_bert_tokens(self):
        """Agrega tokens especiales BERT al tokenizador."""
        from quijote_bert import BERT_SPECIAL
        next_id = max(self.tokenizer.token2id.values()) + 1
        for token, _ in BERT_SPECIAL.items():
            if token not in self.tokenizer.token2id:
                self.tokenizer.token2id[token] = next_id
                self.tokenizer.id2token[next_id] = token
                next_id += 1

    def _train_gpt(self, documents, epochs, batch_size, max_length, grad_accum, base_lr):
        dataset = LMDataset(self.tokenizer, documents, max_length)
        pad_id  = self.tokenizer.pad_id
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_lm(b, pad_id),
            drop_last=False,
        )

        steps_per_epoch = max(len(loader) // grad_accum, 1)
        total_steps     = steps_per_epoch * epochs
        warmup_steps    = max(steps_per_epoch, 50)

        opt = AdamW(self.gpt.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95))
        scheduler = make_cosine_scheduler(opt, warmup_steps, total_steps)

        log.info(f"  LR={base_lr} | warmup={warmup_steps} | total={total_steps} pasos")

        for epoch in range(epochs):
            self.gpt.train()
            total_loss = 0.0
            opt.zero_grad()

            for step, (src, tgt) in enumerate(loader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                logits = self.gpt(src)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=pad_id,
                ) / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), 1.0)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    _free()

            avg = total_loss / max(len(loader), 1)
            lr  = opt.param_groups[0]["lr"]
            log.info(f"  GPT Epoch {epoch+1}/{epochs} | Loss: {avg:.4f} | LR: {lr:.6f}")

        del opt, scheduler
        _free()

    def _train_bert(self, documents, epochs, batch_size, grad_accum, base_lr):
        """Entrena BERT con MLM + NSP."""
        prep = BertDataPrep(self.tokenizer, max_seq=self.bert_cfg.get("max_seq", 256))

        # Tokenizar todos los documentos en oraciones
        all_sentences = []
        for doc in documents:
            sentences = [s.strip() for s in doc.replace('\n', ' ').split('.') if len(s.strip()) > 10]
            for sent in sentences:
                ids = self.tokenizer.encode(sent)
                if ids:
                    all_sentences.append(ids)

        if len(all_sentences) < 2:
            log.warning("  ⚠️  Muy pocas oraciones para BERT, saltando...")
            return

        total_steps  = len(all_sentences) * epochs // max(batch_size * grad_accum, 1)
        warmup_steps = max(total_steps // 10, 20)

        opt = AdamW(self.bert.parameters(), lr=base_lr * 0.5, weight_decay=0.01)
        scheduler = make_cosine_scheduler(opt, warmup_steps, max(total_steps, 1))

        log.info(f"  {len(all_sentences)} oraciones | LR={base_lr*0.5}")

        for epoch in range(epochs):
            self.bert.train()
            total_mlm  = 0.0
            total_nsp  = 0.0
            steps      = 0
            opt.zero_grad()

            # Barajar oraciones cada época
            random.shuffle(all_sentences)

            for i in range(0, len(all_sentences) - batch_size, batch_size):
                batch_sents = all_sentences[i:i + batch_size + 1]
                batch = prep.prepare_batch(batch_sents, batch_size=batch_size)
                if batch is None:
                    continue

                input_t, segment_t, labels_t, nsp_t, pad_mask_t = [
                    t.to(self.device) for t in batch
                ]

                # Loss MLM
                mlm_logits = self.bert.forward_mlm(input_t, segment_t, pad_mask_t)
                mlm_loss = F.cross_entropy(
                    mlm_logits.reshape(-1, mlm_logits.size(-1)),
                    labels_t.reshape(-1),
                    ignore_index=-100,
                )

                # Loss NSP
                nsp_logits = self.bert.forward_nsp(input_t, segment_t, pad_mask_t)
                nsp_loss = F.cross_entropy(nsp_logits, nsp_t)

                # Loss combinado
                loss = (mlm_loss + nsp_loss) / (2 * grad_accum)
                loss.backward()

                total_mlm += mlm_loss.item()
                total_nsp += nsp_loss.item()
                steps += 1

                if steps % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 1.0)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    _free()

            if steps > 0:
                lr = opt.param_groups[0]["lr"]
                log.info(
                    f"  BERT Epoch {epoch+1}/{epochs} | "
                    f"MLM: {total_mlm/steps:.4f} | "
                    f"NSP: {total_nsp/steps:.4f} | "
                    f"LR: {lr:.6f}"
                )

        del opt, scheduler
        _free()

    def _train_embedder(self, documents, epochs, batch_size, max_length, grad_accum, base_lr):
        """Entrena embedder simple (fallback cuando use_bert=False)."""
        pairs, labels = [], []
        for _ in range(800):
            doc = random.choice(documents)
            if len(doc) < 10:
                continue
            mid = len(doc) // 2
            pairs.append((doc[:mid], doc[mid:]))
            labels.append(1.0)
            if len(documents) >= 2:
                d1, d2 = random.sample(documents, 2)
                pairs.append((d1[:100], d2[:100]))
                labels.append(0.0)

        opt = AdamW(self.embedder.parameters(), lr=base_lr * 0.3, weight_decay=0.01)
        self.embedder.train()

        for epoch in range(epochs):
            total_loss = 0.0
            opt.zero_grad()
            indices = list(range(len(pairs)))
            random.shuffle(indices)

            for step, idx in enumerate(indices):
                a_text, b_text = pairs[idx]
                label = torch.tensor([labels[idx]], dtype=torch.float, device=self.device)

                def enc(text):
                    ids = self.tokenizer.encode(text)[:max_length]
                    if not ids:
                        ids = [self.tokenizer.pad_id]
                    t    = torch.tensor([ids], dtype=torch.long, device=self.device)
                    mask = torch.zeros((1, len(ids)), dtype=torch.bool, device=self.device)
                    return self.embedder(t, mask)

                sim  = F.cosine_similarity(enc(a_text), enc(b_text))
                loss = F.mse_loss(sim, label) / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

            log.info(f"  Embedder Epoch {epoch+1}/{epochs} | Loss: {total_loss/max(len(indices),1):.5f}")

        del opt
        _free()

    # ── Índice vectorial ──────────────────────────────────

    def build_index(self, documents: List[str]):
        log.info(f"🔍 Construyendo índice vectorial ({len(documents)} fragmentos)...")
        if self.use_bert and self.bert is not None:
            embeddings = self.bert.encode_texts(
                documents, self.tokenizer, batch_size=32, device=self.device
            )
        else:
            embeddings = self.embedder.encode(
                documents, self.tokenizer, batch_size=32, device=self.device
            )
        self.index.build(embeddings, documents)
        log.info("✅ Índice listo")

    # ── Recuperación ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        if self.use_bert and self.bert is not None:
            q_emb = self.bert.encode_texts([query], self.tokenizer, device=self.device)
        else:
            q_emb = self.embedder.encode([query], self.tokenizer, device=self.device)
        results = self.index.search(q_emb[0], top_k=top_k)
        return [doc for _, doc in results]

    # ── Generación con RAG ────────────────────────────────

    def generate(
        self,
        query:              str,
        max_new_tokens:     int   = 120,
        temperature:        float = 0.7,
        top_k:              int   = 40,
        top_p:              float = 0.9,
        repetition_penalty: float = 1.3,
        use_qa:             bool  = True,
    ) -> str:
        if self.gpt is None or self.tokenizer is None:
            raise RuntimeError("Modelo no entrenado.")

        retrieved = self.retrieve(query)

        # Si BERT está disponible, intentar QA primero
        if use_qa and self.use_bert and self.bert is not None and retrieved:
            best_context = retrieved[0]
            qa_answer = self.bert.answer_question(
                query, best_context, self.tokenizer, self.device
            )
            # Si QA encontró algo útil (más de 2 tokens), usarlo como contexto adicional
            if len(qa_answer.strip()) > 5:
                context = f"{qa_answer}. " + " ".join(retrieved)
            else:
                context = " ".join(retrieved)
        else:
            context = " ".join(retrieved)

        prompt    = f"contexto: {context} pregunta: {query} respuesta:"
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)

        new_ids = self.gpt.generate(
            prompt_ids,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k,
            top_p              = top_p,
            repetition_penalty = repetition_penalty,
            eos_id             = self.tokenizer.eos_id,
            device             = self.device,
        )
        return self.tokenizer.decode(new_ids)

    # ── Agregar / reemplazar documentos ──────────────────

    def add_documents(self, new_documents: List[str]):
        """Agrega nuevos documentos al índice SIN reentrenar."""
        all_docs = list(self.index.documents) + new_documents
        log.info(f"📥 Agregando {len(new_documents)} fragmentos (total: {len(all_docs)})")
        self._reindex(all_docs)
        log.info("✅ Índice actualizado")

    def replace_documents(self, new_documents: List[str]):
        """Reemplaza el índice completamente."""
        log.info(f"🔄 Reemplazando índice con {len(new_documents)} fragmentos...")
        self._reindex(new_documents)
        log.info("✅ Índice reemplazado")

    def _reindex(self, documents: List[str]):
        if self.use_bert and self.bert is not None:
            embeddings = self.bert.encode_texts(documents, self.tokenizer, batch_size=32, device=self.device)
        else:
            embeddings = self.embedder.encode(documents, self.tokenizer, batch_size=32, device=self.device)
        self.index.build(embeddings, documents)
        self.index.save(str(self.save_dir))

    # ── Guardar / cargar ──────────────────────────────────

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.tokenizer: self.tokenizer.save(str(self.save_dir))
        if self.gpt:       self.gpt.save(str(self.save_dir))
        if self.bert:      self.bert.save(str(self.save_dir))
        if self.embedder:  self.embedder.save(str(self.save_dir))
        self.index.save(str(self.save_dir))
        log.info(f"💾 Modelo guardado en {self.save_dir}")

    def load(self) -> bool:
        tok_path  = self.save_dir / "tokenizer.json"
        gpt_path  = self.save_dir / "model.pt"
        bert_path = self.save_dir / "bert.pt"
        emb_path  = self.save_dir / "embedder.pt"

        if not (tok_path.exists() and gpt_path.exists()):
            return False

        self.tokenizer = BPETokenizer.load(str(self.save_dir))
        self.gpt       = QuijoteGPT.load(str(self.save_dir), self.device)

        if bert_path.exists():
            self.bert = QuijoteBERT.load(str(self.save_dir), self.device)
            log.info("  🧠 BERT cargado")
        elif emb_path.exists():
            from quijote_core import QuijoteEmbedder
            self.embedder = QuijoteEmbedder.load(str(self.save_dir), self.device)
            log.info("  📦 Embedder cargado (sin BERT)")

        loaded = self.index.load(str(self.save_dir))
        log.info(f"📂 Modelo cargado | índice={'✅' if loaded else '⚠️'}")
        return True


# ══════════════════════════════════════════════════════════
#  Carga de documentos con overlap
# ══════════════════════════════════════════════════════════

def load_documents_from_folder(
    folder_path: str,
    chunk_size:  int = 300,
    overlap:     int = 60,
) -> List[str]:
    documents = []
    folder = Path(folder_path)
    if not folder.exists():
        log.warning(f"Carpeta '{folder_path}' no encontrada.")
        return documents
    for txt_file in sorted(folder.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            step    = chunk_size - overlap
            chunks  = [
                content[i:i + chunk_size]
                for i in range(0, len(content), step)
                if content[i:i + chunk_size].strip()
            ]
            documents.extend(chunks)
            log.info(f"📄 {txt_file.name}: {len(chunks)} fragmentos")
        except Exception as e:
            log.error(f"Error leyendo {txt_file.name}: {e}")
    log.info(f"📚 Total: {len(documents)} fragmentos")
    return documents