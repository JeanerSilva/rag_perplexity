import re

def corrigir_erros_comuns(pergunta):
    substituicoes = {
        "objeivos": "objetivos",
        "objetivo especificos": "objetivos específicos",
        "objeivo": "objetivo",
        "objeivos especificos": "objetivos específicos"
        # adicione outras conforme aparecerem
    }

    for errado, certo in substituicoes.items():
        pergunta = pergunta.replace(errado, certo)

    return pergunta

def inferir_tipos_relevantes_regex(pergunta):
    pergunta = pergunta.lower()
    tipos = []

    if re.search(r"\bobjetiv[oa]s?\s+específic[oa]s?\b", pergunta):
        tipos.append("objetivo_especifico")
    if re.search(r"\bobjetiv[oa]s?\s+gerais?\b", pergunta):
        tipos.append("objetivo_geral")
    if re.search(r"\bobjetiv[oa]s?\s+estratégic[oa]s?\b", pergunta):
        tipos.append("objetivo_estrategico")
    if re.search(r"\bproblema\b|\bproblemas?\b", pergunta):
        tipos.append("problema")
    if re.search(r"\bcausa\b|\bcausas\b", pergunta):
        tipos.append("causa")
    if re.search(r"\bjustificativa\b|\bpor que\b|\bporque\b", pergunta):
        tipos.append("justificativa")
    if re.search(r"\bevid[êe]ncia\b|\bevid[êe]ncias\b", pergunta):
        tipos.append("evidencia")
    if re.search(r"\bentregas?\b|\bresultados?\b", pergunta):
        tipos.append("entrega")
    if re.search(r"\bp[úu]blico[- ]?alvo\b", pergunta):
        tipos.append("publico_alvo")
    if re.search(r"\bmarco legal\b", pergunta):
        tipos.append("marco_legal")
    if re.search(r"\barticula[çc][ãa]o federativa\b", pergunta):
        tipos.append("articulacao_federativa")

    return tipos or None
pergunta = "quais os objeivos específicos do programa 1144?"
pergunta = corrigir_erros_comuns(pergunta.lower())

print(inferir_tipos_relevantes_regex(pergunta))