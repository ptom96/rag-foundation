Rag-foundation

A from-scratch RAG pipeline over 21 arXiv papers on LLMs, embeddings and retrieval. It answers questions with sentence-level citations (paper ID + page), and every chunking and retrieval choice is measured against a hand-labelled evaluation set.

Built to learn RAG properly: no LangChain or LlamaIndex. Each stage is written by hand and its design decisions are recorded in docs/DESIGN_NOTE.MD. Everything runs locally on free tools.

How it works
PDFs → parse (PyMuPDF) → split Content / References / Appendix → clean
     → chunk (recursive or semantic) → embed (nomic-embed-text-v1.5)
     → store (NumPy + JSON, or Postgres + pgvector)
     → retrieve top-k (cosine) → answer with Llama 3.2 3B (Ollama)
     → validate citations against the retrieved chunks
Parsing: I compared pypdf, pdfplumber and PyMuPDF. PyMuPDF kept word spacing and paragraph structure best.
Section splitting: references and appendices are detected line by line and tagged, so retrieval can filter to body content by default.
Chunking: two strategies:
Recursive: paragraph- and sentence-aware, never breaks a sentence, and keeps headings together. The default size of 600 characters with 150 overlap was chosen from the paragraph and sentence length distributions in the corpus.
Semantic: splits where the embedding similarity between neighbouring sentences drops below a threshold, with a minimum chunk size.
Embedding: nomic-embed-text-v1.5 with search_document: / search_query: prefixes and normalised vectors.
Storage and retrieval: brute-force cosine over a NumPy index, or Postgres + pgvector with an HNSW index. I compared the two and swept ef_search (5–400), with optional paper-level filtering.
Generation: Llama 3.2 3B via Ollama at temperature 0, constrained to a Pydantic JSON schema. Each answer sentence carries its own citations, and the model refuses when the context doesn't contain the answer.
Citation check: every citation is verified to point at a chunk that was actually retrieved.
Evaluation

There are 20 hand-written questions, each with the expected answer, source paper, page and evidence:

8 single-fact
7 multi-hop
3 table/figure
2 not-in-corpus (to test refusal)

src/evaluate.py runs the set and writes each run to results/. Answers are then graded manually as correct, partial or wrong.

Chunking	Size / overlap	Retrieval hit@5*	Correct	Partial	Wrong
Recursive	300 / 75	15 / 18	10	7	3
Semantic	300 / 75	16 / 18	10	7	3
Recursive	1200 / 300	16 / 18	6	6	8

*Hit@5 counts the 18 answerable questions; the 2 not-in-corpus questions are excluded.

Takeaway: larger chunks retrieved the right paper just as often but produced far more wrong answers. With a 3B model, extra context in each chunk dilutes the answer. Small chunks worked best, and semantic chunking matched recursive chunking on answer quality.

Stack

Python · PyMuPDF · sentence-transformers (nomic-embed-text-v1.5) · NumPy · PostgreSQL + pgvector (HNSW) · psycopg · Pydantic · Ollama (Llama 3.2 3B) · pandas

Run it

Prerequisites: Python 3.11+, Ollama with llama3.2:3b pulled, and (for the pgvector path) a local Postgres with the pgvector extension. Put POSTGRES_PASSWORD in a .env file.

bash
pip install -r requirements.txt

# Build the index from PDFs in data/papers (see papers.txt for the list)
python src/build.py --chunk_size 300 --chunk_overlap 75 --run_name C_300_75

# Ask a question
python src/ask.py --query "What problem does Self-RAG address?" --run_name C_300_75

# Run the evaluation set
python src/evaluate.py --run_name my_run

The PDFs aren't committed. Download them from arXiv using the IDs in papers.txt.

Repo layout
src/pipeline.py        parsing, cleaning, chunking, embedding, retrieval, answer, citation check
src/build.py           build an index for one chunking configuration
src/ask.py             ask one question from the command line
src/evaluate.py        run the evaluation set, write results to results/
src/indexing.ipynb     load embeddings into Postgres/pgvector
docs/DESIGN_NOTE.MD    design decisions and reasoning per stage
data/input/            evaluation dataset
results/               one spreadsheet per experiment run
Next steps
Hybrid retrieval (BM25 + vectors) and a re-ranker
Automated evaluation (retrieval metrics plus LLM-as-judge) to replace manual grading
Relative paths and config instead of hard-coded local paths
A small UI (Streamlit) and a deployed version