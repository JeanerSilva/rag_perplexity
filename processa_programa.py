import re
import json
from PyPDF2 import PdfReader
from typing import List
import tiktoken

# Configurações
pdf_path = "Espelho_SIOP_1752688760490.pdf"  # Altere para seu caminho
output_path = "programas_chunked_tokenlimit_300.jsonl"
token_limit = 300

# Tokenizador compatível com modelos OpenAI
enc = tiktoken.get_encoding("cl100k_base")

# Campos que queremos extrair
campos_fixos = [
    "Órgão", "Tipo de Programa", "Objetivos Estratégicos", "Público Alvo", "Problema",
    "Causa do problema", "Evidências do problema", "Justificativa para a intervenção",
    "Evolução histórica", "Comparações Internacionais", "Relação com os ODS",
    "Agentes Envolvidos", "Articulação federativa", "Enfoque Transversal",
    "Marco Legal", "Planos nacionais, setoriais e regionais"
]

# Função para extrair seções
def extract_section(text: str, start_label: str, end_labels: List[str]) -> str:
    pattern = rf"{re.escape(start_label)}\s*:(.*?)(?=" + "|".join([rf"{re.escape(lbl)}\s*:" for lbl in end_labels]) + r"|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Quebra de texto em chunks limitados por token
def split_text_into_chunks(text: str, token_limit: int) -> List[str]:
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

# Ler PDF
reader = PdfReader(pdf_path)
full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
program_blocks = list(re.finditer(r"Programa\s+(\d+)\s*-\s*(.+?)\n", full_text))

# Gerar chunks
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
                "programa_nome": prog_nome,
                "campos_presentes": [campo],
                "texto": sub
            })

# Exportar para JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for item in final_chunks:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Arquivo salvo: {output_path} ({len(final_chunks)} chunks)")
