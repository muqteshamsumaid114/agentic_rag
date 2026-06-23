"""
Hybrid retrieval: dense (FAISS) + sparse (BM25), fused with Reciprocal Rank
Fusion, then re-ranked with a cross-encoder.

Swap-in note: replace `FAISSDenseIndex` with a Weaviate-backed class
(`WeaviateDenseIndex` below, stubbed) if you want a managed vector DB instead
of a local FAISS index. The `HybridRetriever` interface doesn't change.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    doc: Document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: Optional[float] = None


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class FAISSDenseIndex:
    """Dense vector index backed by FAISS (cosine sim via normalized IP)."""

    def __init__(self, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedder = SentenceTransformer(embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents: List[Document] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")

    def add(self, documents: List[Document]) -> None:
        vecs = self._embed([d.text for d in documents])
        self.index.add(vecs)
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        qvec = self._embed([query])
        scores, idxs = self.index.search(qvec, min(top_k, len(self.documents)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append(RetrievedChunk(doc=self.documents[idx], dense_score=float(score)))
        return out

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(str(Path(path) / "index.faiss"))
        with open(Path(path) / "docs.pkl", "rb") as f:
            self.documents = pickle.load(f)


class WeaviateDenseIndex:
    """
    Drop-in replacement for FAISSDenseIndex backed by Weaviate.
    Stubbed to show the integration point — fill in client config for your
    deployment (local docker, Weaviate Cloud, etc).
    """

    def __init__(self, url: str, api_key: Optional[str] = None, class_name: str = "Chunk"):
        import weaviate  # local import so faiss-only setups don't need this dep

        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self.client = weaviate.connect_to_custom(
            http_host=url, http_port=8080, http_secure=False,
            grpc_host=url, grpc_port=50051, grpc_secure=False,
            auth_credentials=auth,
        )
        self.class_name = class_name

    def add(self, documents: List[Document]) -> None:
        collection = self.client.collections.get(self.class_name)
        with collection.batch.dynamic() as batch:
            for d in documents:
                batch.add_object(properties={"text": d.text, "doc_id": d.doc_id, **d.metadata})

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        collection = self.client.collections.get(self.class_name)
        res = collection.query.near_text(query=query, limit=top_k, return_metadata=["score"])
        out = []
        for obj in res.objects:
            doc = Document(doc_id=obj.properties["doc_id"], text=obj.properties["text"])
            out.append(RetrievedChunk(doc=doc, dense_score=float(obj.metadata.score or 0.0)))
        return out


class BM25SparseIndex:
    """Sparse lexical index using BM25Okapi."""

    def __init__(self):
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None

    def add(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        corpus = [_tokenize(d.text) for d in self.documents]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(doc=self.documents[i], sparse_score=float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    dense: List[RetrievedChunk], sparse: List[RetrievedChunk], k: int = 60
) -> List[RetrievedChunk]:
    """Fuse dense + sparse rankings. k is the standard RRF smoothing constant."""
    fused: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense):
        cid = chunk.doc.doc_id
        fused.setdefault(cid, RetrievedChunk(doc=chunk.doc))
        fused[cid].dense_score = chunk.dense_score
        fused[cid].fused_score += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(sparse):
        cid = chunk.doc.doc_id
        fused.setdefault(cid, RetrievedChunk(doc=chunk.doc))
        fused[cid].sparse_score = chunk.sparse_score
        fused[cid].fused_score += 1.0 / (k + rank + 1)

    return sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)


class CrossEncoderReranker:
    """Re-ranks fused candidates with a cross-encoder for precision at the top."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int = 5
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [[query, c.doc.text] for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]


class HybridRetriever:
    """
    Public interface used by agents. Orchestrates:
    dense search -> sparse search -> RRF fusion -> cross-encoder rerank.
    """

    def __init__(
        self,
        dense_index: FAISSDenseIndex,
        sparse_index: BM25SparseIndex,
        reranker: CrossEncoderReranker,
        fusion_pool_size: int = 20,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.fusion_pool_size = fusion_pool_size

    def index(self, documents: List[Document]) -> None:
        self.dense_index.add(documents)
        self.sparse_index.add(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        dense_hits = self.dense_index.search(query, top_k=self.fusion_pool_size)
        sparse_hits = self.sparse_index.search(query, top_k=self.fusion_pool_size)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        candidates = fused[: self.fusion_pool_size]
        return self.reranker.rerank(query, candidates, top_k=top_k)
