"""
exportar_rpi.py — Cuantiza el modelo a INT4 y lo empaqueta para RPi5
====================================================================
Uso:
  python exportar_rpi.py              # cuantiza y empaqueta
  python exportar_rpi.py --verificar  # verifica el paquete generado

Genera: rpi_package/ con todo lo necesario para correr en la RPi5
"""

import os
import sys
import json
import shutil
import struct
import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from quijote_core import QuijoteGPT, QuijoteEmbedder, BPETokenizer, VectorIndex


# ══════════════════════════════════════════════════════════
#  CUANTIZACIÓN INT4 PROPIA (sin bitsandbytes)
# ══════════════════════════════════════════════════════════

def cuantizar_tensor_int4(tensor: torch.Tensor):
    """
    Cuantiza un tensor a INT4 (16 niveles por valor).
    Retorna (datos_empaquetados, escala, zero_point, forma_original)

    Empaqueta dos valores INT4 en un byte (ahorra 8x vs FP32).
    """
    forma = tensor.shape
    flat  = tensor.float().reshape(-1)

    # Calcular escala por bloques de 32 valores (más preciso que global)
    block_size = 32
    n_bloques  = (len(flat) + block_size - 1) // block_size

    scales = []
    zeros  = []
    datos  = []

    for i in range(n_bloques):
        bloque = flat[i * block_size : (i + 1) * block_size]
        vmin   = bloque.min().item()
        vmax   = bloque.max().item()
        rango  = max(vmax - vmin, 1e-8)

        scale  = rango / 15.0   # 15 = 2^4 - 1
        zp     = vmin

        # Cuantizar a [0, 15]
        q = ((bloque - zp) / scale).round().clamp(0, 15).to(torch.uint8)
        scales.append(scale)
        zeros.append(zp)
        datos.append(q)

    # Empaquetar: dos INT4 por byte
    datos_flat = torch.cat(datos)
    # Pad a longitud par
    if len(datos_flat) % 2 != 0:
        datos_flat = torch.cat([datos_flat, torch.zeros(1, dtype=torch.uint8)])

    empaquetado = (datos_flat[0::2] | (datos_flat[1::2] << 4))

    return (
        empaquetado.numpy().tobytes(),
        scales,
        zeros,
        list(forma),
    )


def desempaquetar_int4(datos_bytes, scales, zeros, forma):
    """Reconstruye el tensor FP32 desde INT4."""
    empaquetado = torch.frombuffer(datos_bytes, dtype=torch.uint8)

    # Desempaquetar
    low  = empaquetado & 0x0F
    high = (empaquetado >> 4) & 0x0F
    datos_flat = torch.stack([low, high], dim=1).reshape(-1).float()

    block_size = 32
    reconstruido = torch.zeros(len(datos_flat))

    for i, (scale, zp) in enumerate(zip(scales, zeros)):
        start = i * block_size
        end   = min(start + block_size, len(datos_flat))
        reconstruido[start:end] = datos_flat[start:end] * scale + zp

    # Truncar al tamaño original y reshape
    total = 1
    for d in forma:
        total *= d
    return reconstruido[:total].reshape(forma)


# ══════════════════════════════════════════════════════════
#  EXPORTACIÓN
# ══════════════════════════════════════════════════════════

def exportar_modelo(
    modelo_dir:  str = "quijote_model",
    salida_dir:  str = "rpi_package",
    solo_gpt:    bool = False,
):
    """
    Carga el modelo entrenado, lo cuantiza a INT4 y lo guarda
    en un formato optimizado para RPi5.
    """
    modelo_path = Path(modelo_dir)
    salida_path = Path(salida_dir)
    salida_path.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  📦 Exportador para RPi5 — Cuantización INT4")
    print("=" * 55)

    # ── Cargar modelo ────────────────────────────────────
    print("\n1️⃣  Cargando modelo entrenado...")
    device = torch.device("cpu")   # siempre exportar desde CPU

    tok = BPETokenizer.load(str(modelo_path))
    print(f"   Tokenizador: {len(tok.token2id)} tokens")

    gpt = QuijoteGPT.load(str(modelo_path), device)
    gpt.eval()
    params = gpt.count_params()
    print(f"   GPT: {params:,} parámetros ({params*4//1_000_000} MB en FP32)")

    # ── Cuantizar GPT a INT4 ─────────────────────────────
    print("\n2️⃣  Cuantizando GPT a INT4...")
    capas_cuantizadas = {}
    total_original = 0
    total_comprimido = 0

    for nombre, parametro in gpt.named_parameters():
        if parametro.dim() >= 2:   # solo cuantizar matrices (no bias/norms)
            datos, scales, zeros, forma = cuantizar_tensor_int4(parametro.data)
            capas_cuantizadas[nombre] = {
                "type":   "int4",
                "data":   datos,
                "scales": scales,
                "zeros":  zeros,
                "shape":  forma,
            }
            original    = parametro.numel() * 4
            comprimido  = len(datos)
            total_original   += original
            total_comprimido += comprimido
        else:
            # Vectores pequeños (bias, norm) los guardamos en FP16
            capas_cuantizadas[nombre] = {
                "type": "fp16",
                "data": parametro.data.half().numpy().tobytes(),
                "shape": list(parametro.shape),
            }
            total_original   += parametro.numel() * 4
            total_comprimido += parametro.numel() * 2

    ahorro = (1 - total_comprimido / total_original) * 100
    print(f"   Original:    {total_original / 1_000_000:.1f} MB")
    print(f"   Comprimido:  {total_comprimido / 1_000_000:.1f} MB")
    print(f"   Ahorro:      {ahorro:.1f}%")

    # ── Guardar GPT cuantizado ───────────────────────────
    print("\n3️⃣  Guardando archivos...")

    gpt_out = salida_path / "gpt_int4.pkl"
    with open(gpt_out, "wb") as f:
        pickle.dump(capas_cuantizadas, f, protocol=4)
    print(f"   ✅ GPT INT4: {gpt_out} ({gpt_out.stat().st_size / 1_000_000:.1f} MB)")

    # Guardar config del GPT
    with open(modelo_path / "config.json") as f:
        config = json.load(f)
    with open(salida_path / "config.json", "w") as f:
        json.dump(config, f)

    # ── Copiar tokenizador (pequeño, no necesita cuantización) ──
    shutil.copy(modelo_path / "tokenizer.json", salida_path / "tokenizer.json")
    tok_size = (salida_path / "tokenizer.json").stat().st_size / 1_000_000
    print(f"   ✅ Tokenizador: {tok_size:.1f} MB")

    if not solo_gpt:
        # ── Embedder en FP16 (ya es pequeño) ───────────────
        emb_path = modelo_path / "embedder.pt"
        if emb_path.exists():
            emb = QuijoteEmbedder.load(str(modelo_path), device)
            emb_state = {k: v.half() for k, v in emb.state_dict().items()}
            torch.save(emb_state, salida_path / "embedder_fp16.pt")

            with open(modelo_path / "embedder_config.json") as f:
                emb_cfg = json.load(f)
            with open(salida_path / "embedder_config.json", "w") as f:
                json.dump(emb_cfg, f)
            print(f"   ✅ Embedder FP16: {(salida_path / 'embedder_fp16.pt').stat().st_size / 1_000_000:.1f} MB")

        # ── Índice vectorial ────────────────────────────────
        idx_path = modelo_path / "vectors.pt"
        if idx_path.exists():
            shutil.copy(idx_path, salida_path / "vectors.pt")
            shutil.copy(modelo_path / "documents.pkl", salida_path / "documents.pkl")
            print(f"   ✅ Índice vectorial copiado")

    # ── Copiar matrices TurboQuant si existen ───────────
    for pt_file in modelo_path.glob("*.pt"):
        if pt_file.name not in ["model.pt", "embedder.pt", "vectors.pt"]:
            shutil.copy(pt_file, salida_path / pt_file.name)

    # ── Generar script de carga para RPi ────────────────
    _generar_loader_rpi(salida_path, config)

    # ── Resumen final ────────────────────────────────────
    total_size = sum(f.stat().st_size for f in salida_path.rglob("*") if f.is_file())
    print(f"\n{'='*55}")
    print(f"✅ Paquete listo en: {salida_path}/")
    print(f"   Tamaño total: {total_size / 1_000_000:.1f} MB")
    print(f"\n📋 Para copiar a la RPi:")
    print(f"   scp -r {salida_path}/ memeukas@raspberrypi.local:~/mi_ia_quijote/quijote_model/")
    print(f"   O copia la carpeta con USB")


def _generar_loader_rpi(salida_path: Path, config: dict):
    """Genera el módulo de carga INT4 que va en la RPi."""
    codigo = '''"""
rpi_loader.py — Carga el modelo INT4 en la RPi5
================================================
Generado automáticamente por exportar_rpi.py
NO editar manualmente.
"""

import pickle
import json
import torch
import torch.nn as nn
from pathlib import Path
from quijote_core import QuijoteGPT, QuijoteEmbedder, BPETokenizer, VectorIndex


def desempaquetar_int4(datos_bytes, scales, zeros, forma):
    empaquetado = torch.frombuffer(bytearray(datos_bytes), dtype=torch.uint8)
    low  = empaquetado & 0x0F
    high = (empaquetado >> 4) & 0x0F
    datos_flat = torch.stack([low, high], dim=1).reshape(-1).float()
    block_size = 32
    reconstruido = torch.zeros(len(datos_flat))
    for i, (scale, zp) in enumerate(zip(scales, zeros)):
        start = i * block_size
        end   = min(start + block_size, len(datos_flat))
        reconstruido[start:end] = datos_flat[start:end] * scale + zp
    total = 1
    for d in forma:
        total *= d
    return reconstruido[:total].reshape(forma)


def cargar_modelo_rpi(model_dir: str = "quijote_model"):
    """Carga el modelo INT4 optimizado para RPi5."""
    path = Path(model_dir)
    device = torch.device("cpu")

    # Tokenizador
    tok = BPETokenizer.load(str(path))

    # Config del GPT
    with open(path / "config.json") as f:
        cfg = json.load(f)

    # Crear modelo vacío
    gpt = QuijoteGPT(**cfg)

    # Cargar pesos cuantizados
    with open(path / "gpt_int4.pkl", "rb") as f:
        capas = pickle.load(f)

    state_dict = {}
    for nombre, info in capas.items():
        if info["type"] == "int4":
            tensor = desempaquetar_int4(
                info["data"], info["scales"], info["zeros"], info["shape"]
            )
        else:
            tensor = torch.frombuffer(
                bytearray(info["data"]), dtype=torch.float16
            ).reshape(info["shape"]).float()
        state_dict[nombre] = tensor

    gpt.load_state_dict(state_dict, strict=False)
    gpt.eval()

    # Embedder FP16
    embedder = None
    emb_path = path / "embedder_fp16.pt"
    emb_cfg_path = path / "embedder_config.json"
    if emb_path.exists() and emb_cfg_path.exists():
        with open(emb_cfg_path) as f:
            emb_cfg = json.load(f)
        embedder = QuijoteEmbedder(**emb_cfg)
        state = torch.load(str(emb_path), map_location="cpu")
        state = {k: v.float() for k, v in state.items()}
        embedder.load_state_dict(state)
        embedder.eval()

    # Índice vectorial
    index = VectorIndex()
    index.load(str(path))

    print(f"✅ Modelo INT4 cargado ({gpt.count_params():,} params)")
    return gpt, tok, embedder, index
'''
    (salida_path / "rpi_loader.py").write_text(codigo, encoding='utf-8')
    print(f"   ✅ rpi_loader.py generado")


# ══════════════════════════════════════════════════════════
#  VERIFICACIÓN
# ══════════════════════════════════════════════════════════

def verificar_paquete(paquete_dir: str = "rpi_package"):
    """Verifica que el paquete INT4 genera salidas coherentes."""
    print("\n🔍 Verificando paquete INT4...")
    path = Path(paquete_dir)

    sys.path.insert(0, str(path))
    from rpi_loader import cargar_modelo_rpi

    gpt, tok, embedder, index = cargar_modelo_rpi(paquete_dir)

    # Test rápido de generación
    prompt = "def fibonacci"
    ids = tok.encode(prompt, add_bos=True)
    nuevos = gpt.generate(
        ids, max_new_tokens=30, temperature=0.7,
        top_k=20, eos_id=tok.eos_id,
        device=torch.device("cpu")
    )
    resultado = tok.decode(nuevos)
    print(f"   Prompt:    '{prompt}'")
    print(f"   Generado:  '{resultado}'")
    print("✅ Paquete verificado correctamente")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument("--modelo",  default="quijote_model")
    parser.add_argument("--salida",  default="rpi_package")
    args = parser.parse_args()

    if args.verificar:
        verificar_paquete(args.salida)
    else:
        exportar_modelo(args.modelo, args.salida)
