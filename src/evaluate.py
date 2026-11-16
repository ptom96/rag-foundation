from pipeline import load_index, retrieve, answer, check_citation, retrievepg
import argparse
import pandas as pd
import openpyxl
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--question_path", type=str, default="/Users/tomthomas/Documents/rag-foundation/data/input/evaluation_dataset.xlsx", required=False)
parser.add_argument("--index_path", type=str, default="/Users/tomthomas/Documents/rag-foundation/output", required=False)
parser.add_argument("--top_k", type=int, default=5, required=False)
parser.add_argument("--section_tag", type=str, default="Content")
parser.add_argument("--run_name", type=str, default="stage1")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--ef_search", type=int, default=None)
parser.add_argument("--use_index", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--search_mode", type=str, default="seqsearch")
def parse_doc_id(value: str) -> str | None:
    return None if value.casefold() == "none" else value

parser.add_argument(
    "--paper_list",
    nargs="*",
    type=parse_doc_id,
    default=None,
)
args = parser.parse_args()
ef_search = args.ef_search
use_index = args.use_index
debug = args.debug
search_mode = args.search_mode
paper_list = args.paper_list
question_path = args.question_path
index_path = args.index_path
top_k = args.top_k
section_tag = args.section_tag
refusal_text = "Sorry context doesn't have enough details to answer the question."

path = f"/Users/tomthomas/Documents/rag-foundation/results/{args.run_name}.xlsx"
if os.path.exists(path):
    raise FileExistsError(f"File '{path}' already exists, please rerun with a different run name.")
if search_mode == "seqsearch":
    index, chunks = load_index(index_path, run_name=args.run_name)
questions_df = pd.read_excel(question_path)

if args.limit:
    questions_df = questions_df.head(int(args.limit))
results = []
for idx, row in questions_df.iterrows():
    sentences = []
    citations = []
    question = row["question"]
    expected_answer = row["expected_answer"]
    expected_doc_id = row["doc_id"]
    expected_page_number = row["page_number"]
    expected_type = row["type"]
    evidence = row["evidence"]
    try:
        if search_mode == "seqsearch":
            retrieved_docs = retrieve(question, index, chunks, top_k=top_k, section_tag=section_tag)
        else:
            paper = paper_list[idx] if paper_list else None
            retrieved_docs = retrievepg(question, top_k=top_k, section_tag=section_tag, use_index=use_index, debug=debug, doc_id=paper, ef_search=ef_search)
        if retrieved_docs:
            top_chunk = max(retrieved_docs, key=lambda item: item.score)
        answer_text = answer(question, retrieved_docs)
        citation_check = check_citation(answer_text, retrieved_docs)
    except Exception as e:
        response = f"Error processing question '{question}': {e}"
        results.append({
            "question": question,
            "retrieved_docs": None,
            "top_chunk": None,
            "response": response,
            "hit_top5":"",
            "accuracy":"",
            "notes":"",
            "citations": None,
            "citation_check": "Error",
            "expected_answer": expected_answer,
            "expected_doc_id": expected_doc_id,
            "expected_page_number": expected_page_number,
            "expected_type": expected_type,
            "evidence": evidence
        })
        continue
    
    for sentence in answer_text.sentences:
        response = sentence.text
        sentences.append(response)
        for sentence_citation in sentence.citations:
            citation = sentence_citation
            citations.append(citation)
    results.append({
        "question": question,
        "retrieved_docs": "|".join([retrieved_doc.chunk.doc_id+":"+str(retrieved_doc.chunk.page_number)+":score"+str(retrieved_doc.score) for retrieved_doc in retrieved_docs]),
        "top_chunk": top_chunk.chunk.doc_id+":"+str(top_chunk.chunk.page_number)+":"+top_chunk.chunk.text[:150] +":score"+str(top_chunk.score) if retrieved_docs else None,
        "response": " ".join(sentence for sentence in sentences),
        "hit_top5":"",
        "accuracy":"",
        "notes":"",
        "citations": "|".join([c.doc_id+":"+str(c.page_number) for c in citations]) if citations else None,
        "citation_check": "|".join([f"{status}:{cit.doc_id+':'+str(cit.page_number)}" for status, cit in citation_check]) if citation_check else ("Refusal" if refusal_text in sentences else "Uncited"),
        "expected_answer": expected_answer,
        "expected_doc_id": expected_doc_id,
        "expected_page_number": expected_page_number,
        "expected_type": expected_type,
        "evidence": evidence
    })
results_df = pd.DataFrame(results)
results_df.to_excel(path, index=False)
