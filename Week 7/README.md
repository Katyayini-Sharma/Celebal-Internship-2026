# Week 7: Document Question Answering System (RAG)

**Celebal Technologies Internship — Deep Learning Track**

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that
answers questions grounded in a custom document, rather than relying solely on
a language model's internal (and potentially outdated or hallucinated)
knowledge. The system retrieves relevant chunks from a document and generates
answers using that retrieved context.

## Pipeline

1. **Document Ingestion** – loads a PDF or text file into raw text (`pypdf`)
2. **Text Chunking** – sliding-window word-based chunking with overlap
3. **Embedding Creation** – `sentence-transformers` (`all-MiniLM-L6-v2`) in Colab,
   with an automatic TF-IDF fallback if the Hugging Face Hub isn't reachable
4. **Vector Database** – FAISS (`IndexFlatIP`, cosine similarity)
5. **Query Processing** – embeds the user's question with the same model
6. **Context Retrieval** – top-k similarity search over the vector store
7. **Answer Generation** – `google/flan-t5-base` in Colab, with an extractive
   fallback (returns the most relevant retrieved sentences) when offline
8. **Experiments** – hybrid search (BM25 + vector), lexical re-ranking, and a
   chunk-size comparison

## Dataset

A custom knowledge-base document (`data/knowledge_base.txt`) summarizing the
core deep learning architectures covered in this internship (autoencoders, CNN
vs. ANN, RNN/LSTM/GRU, RAG itself). Swap in your own PDF, resume, notes, or
research paper by changing `DOC_PATH` in the notebook — that's the point of RAG:
answering questions over your own private data.

## Backend Auto-Detection

The notebook checks connectivity to the Hugging Face Hub at runtime:
- **Online (e.g. Google Colab):** uses `sentence-transformers` for embeddings
  and `flan-t5-base` for generation.
- **Offline / restricted network:** automatically falls back to TF-IDF
  embeddings and an extractive answer generator, so the full pipeline still
  runs end-to-end and produces real, grounded output.

The notebook in this repo was executed in an offline sandbox, so its saved
outputs reflect the **TF-IDF / extractive fallback path**. Re-running it in
Google Colab (with internet access) will automatically switch to the
sentence-transformers + flan-t5 path with no code changes required.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Week7_RAG_Document_QA.ipynb
```

Or open directly in Google Colab and run all cells top to bottom.

## Repository Structure

```
.
├── Week7_RAG_Document_QA.ipynb   # Main notebook (executed, with outputs)
├── data/
│   └── knowledge_base.txt         # Sample custom document used for Q&A
├── metrics_report.txt             # Generated system metrics (produced on run)
├── requirements.txt
├── .gitignore
└── README.md
```

## System Metrics (this run)

- Document: 571 words / 3,996 characters
- Chunking: 80-word chunks, 20-word overlap → 10 chunks
- Embedding backend: TF-IDF (scikit-learn), 254-dim (fallback run)
- Vector store: FAISS `IndexFlatIP`
- Generation backend: extractive fallback (fallback run)
- Validated against 6 domain-specific test questions

Full details in `metrics_report.txt`, generated each time the notebook runs.

## Key Learnings

- How retrieval and generation combine to ground language model answers in
  private/custom documents.
- Why chunk size and overlap directly affect retrieval accuracy and context
  completeness.
- How embeddings + a vector database (FAISS) enable fast semantic search.
- How hybrid search (keyword + vector) and re-ranking correct cases where pure
  semantic search misses exact terminology.
- How to design a pipeline that degrades gracefully instead of failing when a
  full model stack isn't available.

## Conclusion

RAG systems like this one are widely used in chatbots, knowledge assistants,
enterprise search, and AI-powered documentation tools — anywhere a language
model needs to answer questions grounded in data it wasn't trained on.
