import re
import json
from PyPDF2 import PdfReader
from typing import List, Dict
import tiktoken

def parse_objetivos_e_entregas(texto: str) -> List[Dict]:
    objetivos = []
    current_objetivo = None

    objetivo_pattern = re.compile(r"Objetivo Específico\s*[:\-]?\s*(\d{4})", re.IGNORECASE)
    entrega_pattern = re.compile(r"Entrega\s*[:\-]?\s*(\d{4})", re.IGNORECASE)

    linhas = texto.splitlines()
    buffer = ""
    current_section = None

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        objetivo_match = objetivo_pattern.match(linha)
        entrega_match = entrega_pattern.match(linha)

        if objetivo_match:
            if current_objetivo:
                current_objetivo["texto"] = buffer.strip()
                objetivos.append(current_objetivo)

            objetivo_id = objetivo_match.group(1)
            current_objetivo = {
                "objetivo_id": objetivo_id,
                "texto": "",
                "entregas": []
            }
            buffer = ""
            current_section = "objetivo"

        elif entrega_match:
            if current_objetivo is None:
                continue

            if current_section == "objetivo" and buffer:
                current_objetivo["texto"] = buffer.strip()
                buffer = ""

            entrega_id = entrega_match.group(1)
            current_section = "entrega"
            current_entrega = {
                "entrega_id": entrega_id,
                "texto": ""
            }
            buffer = ""

        else:
            buffer += linha + " "

            if current_section == "entrega":
                current_entrega["texto"] = buffer.strip()
                if current_objetivo:
                    current_objetivo["entregas"].append(current_entrega)
                    buffer = ""
                    current_section = "objetivo"

    if current_objetivo:
        if current_section == "objetivo":
            current_objetivo["texto"] = buffer.strip()
        objetivos.append(current_objetivo)

    return objetivos


def split_text_into_chunks(text: str, token_limit: int) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(enc.encode(current + s)) <= token_limit:
            current += s + " "
        else:
            if current:
                chunks.append(current.strip())
            current = s + " "
    if current:
        chunks.append(current.strip())
    return chunks


def extract_section(text: str, start_label: str, end_labels: List[str]) -> str:
    pattern = rf"{re.escape(start_label)}\s*:(.*?)(?=" + "|".join([rf"{re.escape(lbl)}\s*:" for lbl in end_labels]) + r"|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "" 

def process_pdf_to_jsonl(pdf_path: str, output_path: str, token_limit: int = 300):
    reader = PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    program_blocks = list(re.finditer(r"Programa\s+(\d+)\s*-\s*(.+?)\n", full_text))


    campos_fixos = [
        "Programa", "Orgão", "Tipo de Programa", "Objetivos Estratégicos", "Público Alvo", "Problema",
        "Causa do problema", "Evidências do problema", "Justificativa para a intervenção",
        "Evolução histórica", "Comparações Internacionais", "Relação com os ODS",
        "Agentes Envolvidos", "Articulação federativa", "Enfoque Transversal",
        "Marco Legal", "Planos nacionais, setoriais e regionais", "Objetivo Geral", "Objetivo Específico", "Objetivos Específicos", 
        "Entregas", "Entrega"
    ]

    final_chunks = []

    for i, match in enumerate(program_blocks):
        prog_codigo = match.group(1).strip()
        prog_nome = match.group(2).strip()
        start = match.end()
        end = program_blocks[i + 1].start() if i + 1 < len(program_blocks) else len(full_text)
        prog_text = full_text[start:end]

        campos_extratos = {}
        for campo in campos_fixos:
            chave = campo.lower().replace(" ", "_")
            campos_extratos[chave] = extract_section(prog_text, campo, campos_fixos)

        for campo, texto in campos_extratos.items():
            if not texto.strip():
                continue
            for sub in split_text_into_chunks(texto, token_limit):
                final_chunks.append({
                    "programa_codigo": prog_codigo,
                    "campos_presentes": campo.lower().replace(" ", "_"),
                    "texto": f"O {campo.lower().replace(" ", "_")} do programa {prog_codigo} - {prog_nome} é: {sub}." 
                })

        # 🧠 Extração de objetivos e entregas estruturadas
        objetivos = parse_objetivos_e_entregas(prog_text)

        for obj in objetivos:
            # Chunk do objetivo
            for sub in split_text_into_chunks(obj["texto"], token_limit):
                final_chunks.append({
                    "programa_codigo": prog_codigo,
                    "campos_presentes": "objetivo_especifico",
                    "objetivo_id": obj["objetivo_id"],
                    "texto": f"{sub} é objetivo específico do programa {prog_codigo}"
                })

            # Chunk das entregas relacionadas
            for entrega in obj["entregas"]:
                for sub in split_text_into_chunks(entrega["texto"], token_limit):
                    final_chunks.append({
                        "programa_codigo": prog_codigo,
                        "campos_presentes": "entrega",
                        "objetivo_id": obj["objetivo_id"],
                        "entrega_id": entrega["entrega_id"],
                        "texto": f"{sub} é entrega do {obj}"
                    })

        # ✅ Chunk auxiliar com nome do programa
        final_chunks.append({
            "programa_codigo": prog_codigo,
            "programa_nome": prog_nome,
            "campos_presentes": "programa_nome",
            "texto": f"O nome do programa {prog_codigo} é: {prog_nome}."
        })

        # ✅ Chunk auxiliar com código do programa
        final_chunks.append({
            "programa_codigo": prog_codigo,
            "programa_nome": prog_nome,
            "campos_presentes": "programa_codigo",
            "texto": f"O código do programa chamado '{prog_nome}' é: {prog_codigo}."
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ JSONL salvo em {output_path} — Total de chunks: {len(final_chunks)}")
    return output_path, len(final_chunks)
