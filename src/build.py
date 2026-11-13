from pipeline import load_pages, split_sections, clean_section, chunk_paragraphs, build_index, semantic_chunking
from pathlib import Path


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, default="/Users/tomthomas/Documents/rag-foundation/data/papers")
parser.add_argument("--chunk_size", type=int, default=600)
parser.add_argument("--chunk_overlap", type=int, default=150)
parser.add_argument("--run_name", type=str, default="stage1")
parser.add_argument("--similarity_threshold", type=float, default=0.64)
parser.add_argument("--min_chunk_size", type=int, default=200)
parser.add_argument("--chunking_strategy", type=str, default="recursive")
args = parser.parse_args()
FOLDER_PATH = args.folder_path
paths = list(Path(FOLDER_PATH).glob("*.pdf"))
chunk_size = args.chunk_size
chunk_overlap = args.chunk_overlap
run_name = args.run_name
similarity_threshold = args.similarity_threshold
min_chunk_size = args.min_chunk_size
chunking_strategy = args.chunking_strategy
print(f"Running build with run_name={run_name}")

grand_total = 0
all_chunks = []
for path in paths:
    doc_id = path.stem
    pages = load_pages(path)
    sections = split_sections(pages)
    section_counts = {"Content": 0, "References": 0, "Appendix": 0}

    for section_name, section_pages in sections.items():
        cleaned_section = clean_section(section_pages)
        if chunking_strategy == "recursive":
            chunks = chunk_paragraphs(cleaned_section, section_name, doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            chunks = semantic_chunking(cleaned_section, section_name, doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size, similarity_threshold=similarity_threshold, debug=False)
        section_counts[section_name] = len(chunks)
        all_chunks.extend(chunks)

    paper_total = sum(section_counts.values())
    grand_total += paper_total
    print(
        f"{doc_id}  content={section_counts['Content']}  "
        f"refs={section_counts['References']}  "
        f"appendix={section_counts['Appendix']}  total={paper_total}"
    )
print(len(all_chunks))
build_index(all_chunks, run_name=run_name)
print(f"Grand total: {grand_total}")