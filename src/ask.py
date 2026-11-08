from pipeline import load_index, retrieve, answer, check_citation

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="What is RAG?")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--index_path", type=str, default="/Users/tomthomas/Documents/rag-foundation/data/index")
parser.add_argument("--section_tag", type=str, default="Content")
args = parser.parse_args()
query = args.query
top_k = args.top_k
index_path = args.index_path
section_tag = args.section_tag

index, chunks = load_index(index_path)
retrieved_docs = retrieve(query, index, chunks, top_k=top_k, section_tag=section_tag)
answer_text = answer(query, retrieved_docs)
citation_check = check_citation(answer_text, retrieved_docs)

for sentence in answer_text.sentences:
    print("Sentence:", sentence.text)
    for sentence_citation in sentence.citations:
        print("Citation:", sentence_citation)

print("Citation Check:", citation_check)
if any(not is_valid for is_valid, _ in citation_check):
    print("Citation check failed for this question.")
if not citation_check:
    print("Refusal, no citations expected.")
else:
    print("All citations were in the retrieved set.")