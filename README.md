# 🗡️ IA Quijotesca

Sistema de IA 100% propio en PyTorch puro — sin HuggingFace, sin APIs, sin internet al correr.  
Entrena en laptop con GPU NVIDIA, corre en Raspberry Pi 5 con 8GB RAM.

---

## 📁 Estructura del proyecto

```
gptai/
├── quijote_core.py        # Motor: tokenizador BPE, Transformer, embedder, índice vectorial
├── quijote_trainer.py     # Entrenamiento + sistema RAG completo
├── turbo_quant.py         # KV cache comprimida (PolarQuant + QJL)
├── chat_quijote.py        # Interfaz de chat principal
├── preparar_datos.py      # Descarga y limpia datos de texto y código
├── exportar_rpi.py        # Cuantiza a INT4 y empaqueta para RPi5
├── actualizar_docs.py     # Agrega/reemplaza documentos sin reentrenar
│
├── datos/                 # Tus libros .txt originales
├── datos_codigo/          # Código Python descargado y propio
├── datos_entrenamiento/   # Generado por preparar_datos.py (lo que entrena el modelo)
│   ├── textos.txt         # Todos los libros unidos y limpios con etiquetas <texto>
│   └── codigo.txt         # Todo el código unido y limpio con etiquetas <codigo>
│
└── quijote_model/         # Generado al entrenar
    ├── config.json        # Configuración del GPT
    ├── model.pt           # Pesos del GPT
    ├── tokenizer.json     # Vocabulario BPE
    ├── embedder_config.json
    ├── embedder.pt        # Pesos del embedder
    ├── vectors.pt         # Índice vectorial
    ├── documents.pkl      # Fragmentos de texto indexados
    ├── polar_R_*.pt       # Matrices TurboQuant (persistentes)
    ├── qjl_P_*.pt
    └── conversaciones/    # Log de chats por fecha
        └── 2026-05-03.txt
```

---

## ⚙️ Arquitectura — todo propio en PyTorch puro

| Componente | Reemplaza | Técnica |
|---|---|---|
| `BPETokenizer` | GPT2Tokenizer (HuggingFace) | Byte-Pair Encoding entrenable |
| `QuijoteGPT` | GPT2LMHeadModel (HuggingFace) | Transformer causal con RoPE + SwiGLU + RMSNorm |
| `QuijoteEmbedder` | SentenceTransformer | Encoder bidireccional + mean pooling |
| `VectorIndex` | FAISS | Similitud coseno en PyTorch puro |
| `KVCacheTurbo` | bitsandbytes | PolarQuant 3 bits + QJL 4 bits |

### Técnicas modernas implementadas
- **RoPE** — Rotary Position Embedding, sin parámetros extra
- **SwiGLU** — activación más eficiente que GELU/ReLU
- **RMSNorm** — más rápido que LayerNorm
- **Weight tying** — embedding y capa de salida comparten pesos (~15% menos RAM)
- **Nucleus sampling** — top-p + top-k + penalización de repetición
- **Gradient accumulation** — simula batches grandes sin usar más RAM

---

## 🖥️ Hardware recomendado

### Para entrenar (laptop)
- **GPU**: NVIDIA RTX 3050 6GB VRAM ✅
- **RAM**: 16GB
- **OS**: Ubuntu Linux (20-30% más rápido que Windows con misma GPU)
- **CUDA**: 12.1+

### Para inferencia (Raspberry Pi 5)
- **RAM**: 8GB
- **Modelo INT4**: ~150MB RAM usado
- **OS**: Raspberry Pi OS 64-bit

---

## 📦 Instalación

### En laptop (Ubuntu/Linux)
```bash
# PyTorch con CUDA para RTX 3050
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### En Raspberry Pi 5
```bash
# Solo PyTorch CPU
pip install torch
```

### Única dependencia
```
torch
```
Nada más. Sin HuggingFace, sin sentence-transformers, sin FAISS, sin bitsandbytes.

---

## 🚀 Flujo de trabajo completo

### 1. Preparar datos
```bash
# Descarga código Python de GitHub + limpia tus libros
python preparar_datos.py todo

# Solo código
python preparar_datos.py codigo

# Solo texto
python preparar_datos.py texto

# Indexar tus propios .py
python preparar_datos.py mios
```

### 2. Verificar cantidad de datos
```bash
# Linux
du -sh datos_entrenamiento/
wc -c datos_entrenamiento/*.txt

# Windows PowerShell
(Get-ChildItem datos_entrenamiento\*.txt | Measure-Object -Property Length -Sum).Sum / 1MB
```

**Cantidad recomendada por tamaño de modelo:**

| Datos | Épocas recomendadas | Loss esperado |
|---|---|---|
| <3 MB | 5-8 | ~5.0 |
| 3-8 MB | 15-25 | ~4.0-4.5 |
| 8-20 MB | 20-30 | ~3.5-4.0 |
| >20 MB | 25-40 | ~3.0-3.5 |

### 3. Entrenar
```bash
python chat_quijote.py
# Primera vez: entrena automáticamente
# Siguientes veces: carga el modelo guardado
```

### 4. Verificar loss del modelo entrenado
```bash
python -c "
import torch
from quijote_core import QuijoteGPT, BPETokenizer
from quijote_trainer import LMDataset, collate_lm, load_documents_from_folder
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok = BPETokenizer.load('quijote_model')
gpt = QuijoteGPT.load('quijote_model', device)
gpt.eval()

docs = load_documents_from_folder('datos_entrenamiento', chunk_size=400)[:50]
ds = LMDataset(tok, docs, max_length=128)
loader = DataLoader(ds, batch_size=2, collate_fn=lambda b: collate_lm(b, tok.pad_id))

total = 0
with torch.no_grad():
    for i, (src, tgt) in enumerate(loader):
        if i >= 10: break
        src, tgt = src.to(device), tgt.to(device)
        logits = gpt(src)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=tok.pad_id)
        total += loss.item()
print(f'Loss actual: {total/10:.4f}')
"
```

**Referencia de loss:**
- `>5.0` → casi no aprendió, reentrenar con más épocas
- `4.0-5.0` → aprendiendo, respuestas básicas
- `3.0-4.0` → bien entrenado, respuestas coherentes ✅
- `<2.5` → sobreentrenamiento, memoriza en lugar de generalizar

### 5. Exportar para RPi5
```bash
# Cuantiza a INT4 (~87% menos RAM en KV cache)
python exportar_rpi.py

# Verificar que funciona
python exportar_rpi.py --verificar

# Copiar a RPi por red
scp -r rpi_package/ pi@raspberrypi.local:~/gptai/quijote_model/

# O por USB
cp -r rpi_package/ /media/tu_usb/
```

### 6. Correr en RPi5
```bash
python chat_quijote.py
# Carga el modelo INT4 automáticamente
# ~150MB RAM usados
```

---

## 📚 Agregar contenido sin reentrenar

```bash
# Ver cuántos fragmentos tiene el índice
python actualizar_docs.py info

# Agregar libros nuevos SIN olvidar los anteriores
cp libro_nuevo.txt datos/
python actualizar_docs.py agregar datos/

# Reemplazar completamente el índice
python actualizar_docs.py reemplazar datos/
```

**El GPT nunca se toca** — solo cambia el índice RAG de búsqueda.

---

## 📖 Fuentes de datos recomendadas

### Libros en español (gratis, dominio público)
- **Proyecto Gutenberg ES**: https://www.gutenberg.org/browse/languages/es
- **Biblioteca Cervantes**: https://www.cervantesvirtual.com
- **Wikisource ES**: https://es.wikisource.org

```bash
# Ejemplos de descarga directa
wget -O datos/quijote.txt "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
wget -O datos/celestina.txt "https://www.gutenberg.org/cache/epub/12901/pg12901.txt"
wget -O datos/lazarillo.txt "https://www.gutenberg.org/cache/epub/38898/pg38898.txt"
wget -O datos/el_principe.txt "https://www.gutenberg.org/cache/epub/1232/pg1232.txt"
wget -O datos/biblia.txt "https://www.gutenberg.org/cache/epub/1108/pg1108.txt"
```

### Código Python (descarga automática)
```bash
python preparar_datos.py codigo
# Descarga ~200 archivos de TheAlgorithms y geekcomputers
# Cubre: ordenamiento, grafos, ML, DP, criptografía, backtracking, etc.
```

---

## ⚙️ Configuración del modelo

En `chat_quijote.py`:

```python
# NANO — ~8M params, ~32MB, RPi con pocos datos
GPT_CONFIG = dict(d_model=128, n_heads=4, n_layers=3, max_seq=256, dropout=0.1)

# PEQUEÑO — ~22M params, ~85MB, RPi5 estándar
GPT_CONFIG = dict(d_model=256, n_heads=8, n_layers=6, max_seq=256, dropout=0.1)

# MEDIANO — ~60M params, ~230MB, RPi5 con muchos datos
GPT_CONFIG = dict(d_model=512, n_heads=8, n_layers=8, max_seq=256, dropout=0.1)

# GRANDE — ~130M params, ~500MB, laptop
GPT_CONFIG = dict(d_model=768, n_heads=12, n_layers=12, max_seq=512, dropout=0.1)  ← actual 300M

# ENORME — ~200M params, ~760MB, laptop con GPU potente
GPT_CONFIG = dict(d_model=1024, n_heads=16, n_layers=12, max_seq=512, dropout=0.1)
```

**Uso de RAM en RPi5 con INT4:**

| Modelo | RAM FP32 | RAM INT4 |
|---|---|---|
| NANO | ~32 MB | ~4 MB |
| PEQUEÑO | ~85 MB | ~11 MB |
| MEDIANO | ~230 MB | ~29 MB |
| GRANDE (actual) | ~500 MB | ~62 MB |

---

## 🔧 TurboQuant — compresión de KV cache

Implementación propia de PolarQuant + QJL (inspirado en paper de Google 2024).

**Cómo funciona:**
1. **PolarQuant**: rota el vector K/V con matriz ortogonal → cuantiza a 3 bits
2. **QJL**: corrige el error residual proyectando con Johnson-Lindenstrauss a 4 bits

**Ahorro de RAM:**
```
Original FP32:    d_head × 32 bits por token
Con TurboQuant:   d_head × 3 + (d_head/4) × 4 ≈ 4 bits
Ahorro:           ~87% menos RAM en KV cache
```

**Importante:** útil principalmente con secuencias largas (500+ tokens). Para secuencias cortas el ahorro es pequeño. Desactivado por defecto en CUDA por conflicto de dispositivos — activar solo en CPU (RPi5).

Para activar en RPi5, en `quijote_model/config.json`:
```json
"use_turbo": true
```

---

## ⚠️ Problemas conocidos y soluciones

### "Input length of input_ids is X, but max_length is Y"
**Causa**: se usaba `max_length` en lugar de `max_new_tokens`  
**Solución**: ya corregido en versión actual — usa `max_new_tokens`

### FutureWarning de torch.load
**Causa**: PyTorch avisa que `weights_only` cambiará en versiones futuras  
**Solución**: ya corregido — usa `weights_only=False` explícito  
**No afecta** el funcionamiento

### Modelo se cuelga sin responder
**Causa**: TurboQuant activo con CUDA — conflicto de dispositivos  
**Solución**: cambiar `"use_turbo": false` en `quijote_model/config.json`

### Missing keys: `blocks.X.attn.rope` / Unexpected: `blocks.X.attn._rope`
**Causa**: modelo entrenado con versión anterior del código (atributo renombrado)  
**Solución**: ya corregido en `QuijoteGPT.load()` — renombra claves automáticamente

### Loss no baja de 6.0
**Causa**: LR muy bajo (TurboQuantOptimizer lo bajaba a 0.000001)  
**Solución**: ya corregido — LR fijo con scheduler coseno, nunca baja de 1e-4

### La laptop se suspende durante entrenamiento
```bash
# Linux — evitar suspensión
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Revertir después
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Windows PowerShell
powercfg /change standby-timeout-ac 0
```

---

## 📋 Comandos rápidos de referencia

```bash
# Preparar datos
python preparar_datos.py todo

# Entrenar / chatear
python chat_quijote.py

# Ver loss actual del modelo
python -c "..." # (ver sección verificar loss)

# Agregar documentos sin reentrenar
python actualizar_docs.py agregar datos/
python actualizar_docs.py reemplazar datos/
python actualizar_docs.py info

# Exportar para RPi5
python exportar_rpi.py
python exportar_rpi.py --verificar

# Reentrenar desde cero
rm -rf quijote_model/
python chat_quijote.py

# Evitar suspensión en Linux
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

---

## 🗂️ Historial de versiones

| Versión | Cambios |
|---|---|
| v1 | Sistema inicial con HuggingFace + FAISS |
| v2 | Reemplazo completo por PyTorch puro — BPE, GPT, embedder, VectorIndex |
| v3 | TurboQuant real (PolarQuant + QJL) para KV cache |
| v4 | Fix LR (TurboQuantOptimizer corregido) + scheduler coseno real |
| v5 | Soporte código Python + modelo 300M + exportación INT4 para RPi5 |
| v6 | Fix `_rope→rope` compatibilidad + `use_turbo=False` por defecto en CUDA |

---

*Sistema desarrollado 100% en Python + PyTorch puro. Sin dependencias externas de IA.*
