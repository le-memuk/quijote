"""
chat_quijote.py — IA con GPT + BERT — texto + código Python
============================================================
"""

import datetime
import torch
from pathlib import Path
from quijote_trainer import QuijoteRAG, load_documents_from_folder

# ═══════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════

CARPETA      = "datos_entrenamiento"
SAVE_DIR     = "quijote_model"
BPE_VOCAB    = 10000

# GPT — genera respuestas
GPT_CONFIG = dict(d_model=512, n_heads=8, n_layers=8, max_seq=256, dropout=0.1)

# BERT — entiende preguntas y encuentra respuestas exactas
# Más ligero que el GPT intencionalmente
BERT_CONFIG = dict(d_model=256, n_heads=8, n_layers=6, max_seq=256, dropout=0.1)

EPOCAS_GPT  = 6
EPOCAS_BERT = 4   # BERT necesita menos épocas que GPT
BATCH       = 4
GRAD_ACCUM  = 2
BASE_LR     = 1e-3
MAX_LENGTH  = 256

# ═══════════════════════════════════════════════════


def detectar_modo(pregunta: str) -> str:
    palabras_codigo = [
        "código", "codigo", "función", "funcion", "clase", "def ",
        "python", "script", "programa", "bucle", "loop", "lista",
        "diccionario", "error", "bug", "print", "return", "import",
        "cómo hago", "como hago", "escribe", "genera", "ejemplo de"
    ]
    for palabra in palabras_codigo:
        if palabra in pregunta.lower():
            return "codigo"
    return "texto"


def formatear_respuesta(respuesta: str, modo: str) -> str:
    respuesta = respuesta.strip()
    if not respuesta:
        return "(sin respuesta)"
    if modo == "codigo":
        if any(kw in respuesta for kw in ["def ", "class ", "import ", "print(", "return "]):
            return f"\n```python\n{respuesta}\n```"
    return respuesta


def registrar_sesion(pregunta: str, respuesta: str, modo: str):
    hoy     = datetime.date.today().isoformat()
    log_dir = Path(SAVE_DIR) / "conversaciones"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{hoy}.txt", "a", encoding="utf-8") as f:
        ahora = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{ahora}][{modo}] Tú: {pregunta}\n")
        f.write(f"[{ahora}][{modo}] IA:  {respuesta}\n\n")


def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 55)
    print("  🗡️  IA Quijotesca — GPT + BERT — texto + código")
    print(f"  🖥️  Dispositivo: {device_name}")
    print("=" * 55)

    carpeta = Path(CARPETA)
    if not carpeta.exists() or not list(carpeta.glob("*.txt")):
        print(f"\n⚠️  No hay datos en '{CARPETA}/'")
        print("   Ejecuta primero: python preparar_datos.py todo")
        return

    rag = QuijoteRAG(
        save_dir       = SAVE_DIR,
        bpe_vocab_size = BPE_VOCAB,
        gpt_config     = GPT_CONFIG,
        bert_config    = BERT_CONFIG,
        use_bert       = True,
    )

    marca        = Path(SAVE_DIR) / ".archivos_usados"
    txts         = sorted(carpeta.glob("*.txt"))
    lista_actual = sorted(p.name for p in txts)

    if rag.load():
        lista_anterior = marca.read_text().splitlines() if marca.exists() else []
        if lista_actual != lista_anterior:
            print("\n📋 Detecté cambios en datos — actualizando índice...")
            docs = load_documents_from_folder(CARPETA, chunk_size=400, overlap=80)
            rag.replace_documents(docs)
            marca.write_text("\n".join(lista_actual))
            print(f"✅ Índice actualizado con {len(docs)} fragmentos\n")
        else:
            bert_status = "🧠 BERT activo" if rag.bert is not None else "📦 Embedder"
            print(f"📂 Modelo cargado ({len(rag.index.documents)} fragmentos) | {bert_status}\n")
    else:
        print(f"\n🚀 Entrenando GPT + BERT desde cero...")
        print(f"   GPU: {device_name}")
        print(f"   Esto puede tardar 1-3 horas.\n")

        docs = load_documents_from_folder(CARPETA, chunk_size=400, overlap=80)
        rag.train(
            docs,
            epochs_gpt  = EPOCAS_GPT,
            epochs_bert = EPOCAS_BERT,
            batch_size  = BATCH,
            max_length  = MAX_LENGTH,
            grad_accum  = GRAD_ACCUM,
            base_lr     = BASE_LR,
        )
        rag.build_index(docs)
        rag.save()
        marca.write_text("\n".join(lista_actual))
        print("\n💾 Modelo guardado.")
        print("   Para exportar a RPi5: python exportar_rpi.py\n")

    print("🤖 IA lista. Escribe tu pregunta o 'salir' para terminar.")
    print("   GPT genera respuestas | BERT encuentra respuestas exactas")
    print("   Conversaciones en quijote_model/conversaciones/\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not pregunta:
            continue
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break

        try:
            modo      = detectar_modo(pregunta)
            respuesta = rag.generate(
                pregunta,
                max_new_tokens     = 200 if modo == "codigo" else 150,
                temperature        = 0.5 if modo == "codigo" else 0.7,
                top_k              = 30,
                top_p              = 0.9,
                repetition_penalty = 1.3,
                use_qa             = True,   # BERT busca respuesta exacta primero
            )
            respuesta_fmt = formatear_respuesta(respuesta, modo)
            print(f"IA: {respuesta_fmt}\n")
            registrar_sesion(pregunta, respuesta, modo)
        except Exception as e:
            print(f"⚠️  Error: {e}\n")

    print("👋 ¡Hasta luego!")


if __name__ == "__main__":
    main()