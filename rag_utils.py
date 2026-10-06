import json
import re
from math import sqrt


class RAG:
    def __init__(
        self,
        raw_data,
        use_custom_text_preprocessing,
        chunking=None,
        retriever_type="bm25",
        embedding_model="text-embedding-3-small",
        k=5,
        generation_model="gpt-4.1-mini",
    ):
        self.raw_data = raw_data
        self.raw_text = self._coerce_raw_text(raw_data)
        self.use_custom_text_preprocessing = use_custom_text_preprocessing
        self.chunking = self._resolve_chunking_config(chunking)
        self.chunking_method = self.chunking["chunking_method"]
        self.chunk_size_words = self.chunking.get("chunk_size_words")
        self.chunk_overlap_words = self.chunking.get("chunk_overlap_words")
        self.retriever_type = retriever_type
        self.embedding_model = embedding_model
        self.k = k
        self.generation_model = generation_model

        self.clean_text = None
        self.main_law_text = None
        self.transitory_chunk = ""
        self.articles = []
        self.chunks = []
        self.retriever = None
        self.tokenized_chunks = []
        self.chunk_embeddings = []

    def _coerce_raw_text(self, raw_data):
        if isinstance(raw_data, str):
            return raw_data
        return "\n".join(raw_data)

    def _resolve_chunking_config(self, chunking):
        if chunking is None:
            chunking = {
                "chunking_method": "article_custom",
                "chunk_size_words": 250,
                "chunk_overlap_words": 50,
            }

        if "chunking_method" not in chunking:
            raise ValueError("chunking must contain the key 'chunking_method'")

        chunking_method = chunking["chunking_method"]
        if chunking_method is None:
            chunking_method = "article_custom"
            chunking = {
                **chunking,
                "chunking_method": chunking_method,
            }

        if chunking_method == "article_custom":
            chunking.setdefault("chunk_size_words", 250)
            chunking.setdefault("chunk_overlap_words", 50)
            self._validate_size_overlap_config(chunking)
            return chunking

        if chunking_method == "size_overlap":
            chunking.setdefault("chunk_size_words", 250)
            chunking.setdefault("chunk_overlap_words", 50)
            self._validate_size_overlap_config(chunking)
            return chunking

        if chunking_method == "semantic":
            chunking.setdefault("similarity_threshold", 0.4)
            chunking.setdefault("sentence_window_size", 1)
            self._validate_semantic_config(chunking)
            return chunking

        raise ValueError(f"Unsupported chunking_method: {chunking_method}")

    def _validate_size_overlap_config(self, chunking):
        chunk_size_words = chunking["chunk_size_words"]
        chunk_overlap_words = chunking["chunk_overlap_words"]

        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be greater than 0")

        if chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words must be greater than or equal to 0")

        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    def _validate_semantic_config(self, chunking):
        similarity_threshold = chunking["similarity_threshold"]
        sentence_window_size = chunking["sentence_window_size"]

        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if sentence_window_size <= 0:
            raise ValueError("sentence_window_size must be greater than 0")

    def _normalize_whitespace(self, text):
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def preprocess_text(self):
        if self.use_custom_text_preprocessing:
            return self._preprocess_legal_text()

        self.clean_text = self._normalize_whitespace(self.raw_text)
        self.main_law_text = self.clean_text
        self.transitory_chunk = ""
        return self.clean_text

    def _preprocess_legal_text(self):
        official_header_pattern = (
            r"(?m)^\s*(?:\d+\s+)?\(Primera Sección\)\s+DIARIO OFICIAL\s+"
            r"Miércoles 17 de octubre de 2012(?:\s+\d+)?\s*$|"
            r"^\s*Miércoles 17 de octubre de 2012\s+DIARIO OFICIAL\s+"
            r"\(Primera Sección\)(?:\s+\d+)?\s*$"
        )
        page_marker_pattern = r"(?m)^\s*--\s*\d+\s+of\s+\d+\s*--\s*$"

        clean_text = re.sub(official_header_pattern, "", self.raw_text)
        clean_text = re.sub(page_marker_pattern, "", clean_text)
        clean_text = self._normalize_whitespace(clean_text)

        paragraphs = []
        current_paragraph = []
        for line in clean_text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(stripped)
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        clean_text = "\n\n".join(paragraphs)

        transitory_match = re.search(
            r"(?m)^(?:TRANSITORIOS?|Transitorios?)\b",
            clean_text,
        )
        if transitory_match:
            main_law = clean_text[: transitory_match.start()].strip()
            transitory = clean_text[transitory_match.start() :].strip()
        else:
            main_law = clean_text
            transitory = ""

        self.clean_text = clean_text
        self.main_law_text = main_law
        self.transitory_chunk = transitory
        return self.clean_text

    def create_chunks(self, client=None):
        if self.clean_text is None:
            self.preprocess_text()

        if self.chunking_method == "article_custom":
            self.chunks = self._chunk_by_articles()
        elif self.chunking_method == "size_overlap":
            self.chunks = self._chunk_by_size_overlap(self.clean_text)
        elif self.chunking_method == "semantic":
            self.chunks = self._chunk_semantically(client=client)
        else:
            raise ValueError(f"Unsupported chunking_method: {self.chunking_method}")

        return self.chunks

    def _chunk_by_articles(self):
        text = self.main_law_text or self.clean_text
        article_pattern = r"(?m)^(?:Art[ií]culo\s+\d+\.?\s*[-–—]?)"
        splits = [m.start() for m in re.finditer(article_pattern, text)]

        articles = []
        if splits:
            preamble = text[: splits[0]].strip()
            if preamble:
                articles.append(preamble)
            for i in range(len(splits)):
                start = splits[i]
                end = splits[i + 1] if i + 1 < len(splits) else len(text)
                art_text = text[start:end].strip()
                if art_text:
                    articles.append(art_text)
        else:
            articles = [text]

        self.articles = articles

        chunks = []
        for art in articles:
            words = art.split()
            if len(words) <= self.chunk_size_words:
                chunks.append(art)
            else:
                chunks.extend(self._chunk_by_size_overlap(art))

        if self.transitory_chunk:
            trans_words = self.transitory_chunk.split()
            if len(trans_words) <= self.chunk_size_words:
                chunks.append(self.transitory_chunk)
            else:
                chunks.extend(self._chunk_by_size_overlap(self.transitory_chunk))

        return chunks

    def _chunk_by_size_overlap(self, text):
        words = text.split()
        if not words:
            return []

        step = self.chunk_size_words - self.chunk_overlap_words
        chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size_words]
            chunks.append(" ".join(chunk_words))
            if start + self.chunk_size_words >= len(words):
                break
        return chunks

    def _split_sentences(self, text):
        sentence_pattern = (
            r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿])|(?<=\.)\s+(?=Artículo\s+\d+)"
        )
        sentences = re.split(sentence_pattern, text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _group_sentences_by_window(self, sentences):
        window_size = self.chunking["sentence_window_size"]
        if window_size <= 1:
            return sentences

        grouped_sentences = []
        for start in range(0, len(sentences), window_size):
            grouped_sentences.append(" ".join(sentences[start : start + window_size]))
        return grouped_sentences

    def _chunk_semantically(self, client=None):
        if client is None:
            raise ValueError("client is required when chunking_method='semantic'")

        self.articles = []
        sentences = self._split_sentences(self.clean_text)
        docs = self._group_sentences_by_window(sentences)

        if not docs:
            return []
        if len(docs) == 1:
            return docs

        embeddings = self._embed_texts_with_openai(
            texts=docs,
            client=client,
        )
        threshold = self.chunking["similarity_threshold"]
        chunks = [[docs[0]]]

        for index in range(1, len(docs)):
            similarity = self._cosine_similarity_from_vectors(
                embeddings[index - 1],
                embeddings[index],
            )
            if similarity >= threshold:
                chunks[-1].append(docs[index])
            else:
                chunks.append([docs[index]])

        return [" ".join(chunk) for chunk in chunks]

    def tokenize_for_bm25(self, text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def build_retriever(self, client=None):
        if not self.chunks:
            self.create_chunks(client=client)

        if self.retriever_type == "bm25":
            return self._build_bm25_retriever()
        if self.retriever_type == "openai_embeddings":
            return self._build_openai_embeddings_retriever(client)

        raise ValueError(f"Unsupported retriever_type: {self.retriever_type}")

    def _build_bm25_retriever(self):
        from rank_bm25 import BM25Okapi

        self.tokenized_chunks = [
            self.tokenize_for_bm25(chunk) for chunk in self.chunks
        ]
        self.retriever = BM25Okapi(self.tokenized_chunks)
        return self.retriever

    def _build_openai_embeddings_retriever(self, client):
        if client is None:
            raise ValueError(
                "client is required when retriever_type='openai_embeddings'"
            )

        self.chunk_embeddings = self._embed_texts_with_openai(
            texts=self.chunks,
            client=client,
        )
        self.retriever = {
            "type": "openai_embeddings",
            "embedding_model": self.embedding_model,
        }
        return self.retriever

    def _embed_texts_with_openai(self, texts, client, batch_size=64):
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)

        return [self._normalize_vector(embedding) for embedding in embeddings]

    def _normalize_vector(self, vector):
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _cosine_similarity_from_vectors(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def retrieve(self, question, k=None, client=None):
        self.build_retriever(client=client)
        k = k or self.k

        if self.retriever_type == "bm25":
            return self._retrieve_bm25(question, k)
        if self.retriever_type == "openai_embeddings":
            return self._retrieve_openai_embeddings(question, k, client)

        raise ValueError(f"Unsupported retriever_type: {self.retriever_type}")

    def _retrieve_bm25(self, question, k):
        tokenized_question = self.tokenize_for_bm25(question)
        scores = self.retriever.get_scores(tokenized_question)
        top_indices = sorted(
            range(len(scores)),
            key=lambda chunk_index: scores[chunk_index],
            reverse=True,
        )[:k]

        retrieved_chunks = []
        for rank, chunk_index in enumerate(top_indices, start=1):
            retrieved_chunks.append({
                "rank": rank,
                "chunk_index": int(chunk_index),
                "score": float(scores[chunk_index]),
                "chunk": self.chunks[chunk_index],
            })
        return retrieved_chunks

    def _retrieve_openai_embeddings(self, question, k, client):
        if client is None:
            raise ValueError(
                "client is required when retriever_type='openai_embeddings'"
            )

        question_embedding = self._embed_texts_with_openai(
            texts=[question],
            client=client,
        )[0]

        scored_chunks = []
        for chunk_index, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = self._cosine_similarity_from_vectors(
                question_embedding,
                chunk_embedding,
            )
            distance = 1 - similarity
            scored_chunks.append((chunk_index, similarity, distance))

        top_results = sorted(
            scored_chunks,
            key=lambda item: item[2],
        )[:k]

        retrieved_chunks = []
        for rank, (chunk_index, similarity, distance) in enumerate(
            top_results, start=1
        ):
            retrieved_chunks.append({
                "rank": rank,
                "chunk_index": int(chunk_index),
                "score": float(similarity),
                "distance": float(distance),
                "chunk": self.chunks[chunk_index],
            })
        return retrieved_chunks

    def build_context(self, retrieved_chunks):
        return "\n\n---\n\n".join(item["chunk"] for item in retrieved_chunks)

    def build_messages(self, question, retrieved_context):
        return [
            {
                "role": "system",
                "content": (
                    "Responde preguntas sobre la Ley Federal para la Prevención e "
                    "Identificación de Operaciones con Recursos de Procedencia Ilícita. "
                    "Usa únicamente el contexto recuperado. Si la respuesta no está en "
                    "el contexto, di que no tienes suficiente información. "
                    "Responde de forma clara, breve y cita el artículo relevante cuando aparezca en el contexto."
                ),
            },
            {
                "role": "user",
                "content": f"Contexto:\n{retrieved_context}\n\nPregunta:\n{question}",
            },
        ]

    def generate_answer(self, question, client, k=None):
        retrieved_chunks = self.retrieve(question, k=k, client=client)
        retrieved_context = self.build_context(retrieved_chunks)
        messages = self.build_messages(question, retrieved_context)

        response = client.chat.completions.create(
            model=self.generation_model,
            messages=messages,
        )

        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_context": retrieved_context,
            "messages": messages,
        }

    def summary(self):
        return {
            "use_custom_text_preprocessing": self.use_custom_text_preprocessing,
            "chunking": self.chunking,
            "chunking_method": self.chunking_method,
            "retriever_type": self.retriever_type,
            "embedding_model": self.embedding_model,
            "k": self.k,
            "generation_model": self.generation_model,
            "clean_characters": len(self.clean_text) if self.clean_text else 0,
            "articles": len(self.articles),
            "chunks": len(self.chunks),
        }


class EVALS:
    def __init__(self, client, evaluation_model="gpt-4.1-mini"):
        self.client = client
        self.evaluation_model = evaluation_model
        self.label_scores = {
            "GOOD": 2,
            "REGULAR": 1,
            "BAD": 0,
        }

    def evaluate_answer_relevancy(self, rag_results):
        import pandas as pd

        evaluations = []
        for result in rag_results:
            question = result["question"]
            answer = result["answer"]
            eval_json = self._run_json_eval(
                system_prompt=self._answer_relevancy_prompt(),
                user_prompt=(
                    f"Pregunta:\n{question}\n\n"
                    f"Respuesta:\n{answer}"
                ),
            )
            evaluations.append({
                "question_number": result.get("question_number"),
                "question": question,
                "answer": answer,
                "answer_relevancy_label": eval_json["label"],
                "answer_relevancy_score": self._label_to_score(eval_json["label"]),
                "answer_relevancy_reasoning": eval_json["reasoning"],
            })
        return pd.DataFrame(evaluations)

    def evaluate_faithfulness(self, rag_results):
        import pandas as pd

        evaluations = []
        for result in rag_results:
            question = result["question"]
            answer = result["answer"]
            retrieved_context = self._get_retrieved_context(result)
            eval_json = self._run_json_eval(
                system_prompt=self._faithfulness_prompt(),
                user_prompt=(
                    f"Pregunta:\n{question}\n\n"
                    f"Contexto recuperado:\n{retrieved_context}\n\n"
                    f"Respuesta:\n{answer}"
                ),
            )
            evaluations.append({
                "question_number": result.get("question_number"),
                "question": question,
                "answer": answer,
                "faithfulness_label": eval_json["label"],
                "faithfulness_score": self._label_to_score(eval_json["label"]),
                "faithfulness_reasoning": eval_json["reasoning"],
            })
        return pd.DataFrame(evaluations)

    def evaluate_context_relevancy(self, rag_results):
        import pandas as pd

        evaluations = []
        for result in rag_results:
            question = result["question"]
            retrieved_context = self._get_retrieved_context(result)
            retrieved_chunk_indexes = [
                item["chunk_index"] for item in result["retrieved_chunks"]
            ]
            eval_json = self._run_json_eval(
                system_prompt=self._context_relevancy_prompt(),
                user_prompt=(
                    f"Pregunta:\n{question}\n\n"
                    f"Contexto recuperado:\n{retrieved_context}"
                ),
            )
            evaluations.append({
                "question_number": result.get("question_number"),
                "question": question,
                "retrieved_chunk_indexes": retrieved_chunk_indexes,
                "context_relevancy_label": eval_json["label"],
                "context_relevancy_score": self._label_to_score(eval_json["label"]),
                "context_relevancy_reasoning": eval_json["reasoning"],
            })
        return pd.DataFrame(evaluations)

    def evaluate_all(self, rag_results):
        answer_relevancy_df = self.evaluate_answer_relevancy(rag_results)
        faithfulness_df = self.evaluate_faithfulness(rag_results)
        context_relevancy_df = self.evaluate_context_relevancy(rag_results)

        combined_df = (
            answer_relevancy_df.merge(
                faithfulness_df[
                    [
                        "question_number",
                        "faithfulness_label",
                        "faithfulness_score",
                        "faithfulness_reasoning",
                    ]
                ],
                on="question_number",
                how="left",
            ).merge(
                context_relevancy_df[
                    [
                        "question_number",
                        "retrieved_chunk_indexes",
                        "context_relevancy_label",
                        "context_relevancy_score",
                        "context_relevancy_reasoning",
                    ]
                ],
                on="question_number",
                how="left",
            )
        )

        summary_df = self.summarize_scores(combined_df)

        return {
            "answer_relevancy": answer_relevancy_df,
            "faithfulness": faithfulness_df,
            "context_relevancy": context_relevancy_df,
            "combined": combined_df,
            "summary": summary_df,
        }

    def summarize_labels(self, eval_df, label_column):
        return (
            eval_df[label_column]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )

    def summarize_scores(self, combined_df):
        import pandas as pd

        columns = {
            "answer_relevancy": "answer_relevancy_score",
            "faithfulness": "faithfulness_score",
            "context_relevancy": "context_relevancy_score",
        }

        rows = []
        for eval_type, score_column in columns.items():
            average_score = float(combined_df[score_column].mean())
            rows.append({
                "eval_type": eval_type,
                "average_score": average_score,
                "normalized_score": average_score / 2.0,
            })

        summary_df = pd.DataFrame(rows)
        overall_average = float(summary_df["average_score"].mean())
        overall_normalized = float(summary_df["normalized_score"].mean())

        overall_row = pd.DataFrame([
            {
                "eval_type": "overall",
                "average_score": overall_average,
                "normalized_score": overall_normalized,
            }
        ])

        return pd.concat([summary_df, overall_row], ignore_index=True)

    def _label_to_score(self, label):
        return self.label_scores.get(label.strip().upper(), 0)

    def _get_retrieved_context(self, result):
        if "retrieved_context" in result and result["retrieved_context"]:
            return result["retrieved_context"]
        return "\n\n---\n\n".join(
            item["chunk"] for item in result.get("retrieved_chunks", [])
        )

    def _run_json_eval(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.evaluation_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    def _answer_relevancy_prompt(self):
        return """
Evalúa la relevancia de la respuesta respecto a la pregunta.
La relevancia mide si la respuesta responde directamente lo que se pregunta.
No evalúes si la respuesta es verdadera ni si está respaldada por un contexto; evalúa únicamente si responde la pregunta.

Usa una de estas tres etiquetas:
"GOOD" - La respuesta responde de forma directa, completa y clara a la pregunta planteada.
"REGULAR" - La respuesta aborda el tema general de la pregunta, pero es parcial, evasiva, redundante o introduce divagaciones innecesarias.
"BAD" - La respuesta no responde la pregunta, cambia de tema o es completamente irrelevante.

Devuelve únicamente un JSON válido con esta estructura:
{
  "label": "GOOD",
  "reasoning": "explicación breve"
}
""".strip()

    def _faithfulness_prompt(self):
        return """
Evalúa la fidelidad (faithfulness) de la respuesta con respecto al contexto recuperado.
La fidelidad mide si todas las afirmaciones fácticas de la respuesta provienen estrictamente del contexto recuperado y no de conocimiento externo ni alucinaciones.

Usa una de estas tres etiquetas:
"GOOD" - Todas las afirmaciones de la respuesta están plenamente sustentadas por el contexto recuperado.
"REGULAR" - Algunas afirmaciones están apoyadas por el contexto, pero la respuesta también contiene afirmaciones no sustentadas, exageradas, ambiguas o que requieren información externa.
"BAD" - La respuesta contradice el contexto, inventa información, o la mayor parte de sus afirmaciones no puede verificarse en el contexto recuperado.

Caso especial:
Si la respuesta dice que no hay suficiente información para responder, y el contexto recuperado realmente no contiene la información necesaria, evalúala como "GOOD".
Si la respuesta dice que no hay suficiente información, pero el contexto sí contiene claramente la respuesta, evalúala como "BAD" o "REGULAR", según qué tan grave sea el caso.

Devuelve únicamente un JSON válido con esta estructura:
{
  "label": "GOOD",
  "reasoning": "explicación breve"
}
""".strip()

    def _context_relevancy_prompt(self):
        return """
Evalúa la relevancia del contexto recuperado respecto a la pregunta.
La relevancia del contexto mide si los chunks recuperados contienen información útil para responder la pregunta.
No evalúes la respuesta generada.
No uses conocimiento externo.
No evalúes si el modelo respondió bien o mal.
Evalúa únicamente si el contexto recuperado permite responder la pregunta.

Usa una de estas tres etiquetas:
"GOOD" - El contexto recuperado contiene información suficiente para responder directamente la pregunta.
"REGULAR" - El contexto recuperado contiene información relacionada, pero falta algún dato clave, la respuesta está incompleta, o solo puede inferirse indirectamente.
"BAD" - El contexto recuperado no contiene información útil para responder la pregunta.

Caso importante:
Si al menos uno de los chunks contiene claramente la información necesaria para responder la pregunta, evalúa el contexto como "GOOD", aunque los demás chunks contengan ruido o información menos útil.

Devuelve únicamente un JSON válido con esta estructura:
{
  "label": "GOOD",
  "reasoning": "explicación breve"
}
""".strip()
