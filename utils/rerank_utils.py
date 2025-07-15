from sentence_transformers import CrossEncoder

def rerank_docs(query, docs, model_name, top_k=5):
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]
