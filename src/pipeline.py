import pandas as pd
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Literal
from ollama import chat
import pymupdf
import os
import re
import numpy as np
import json

FOLDER_PATH = "/Users/tomthomas/Documents/rag-foundation/data/papers"
paths = list(Path(FOLDER_PATH).glob("*.pdf"))
OUTPUT_PATH = "/Users/tomthomas/Documents/rag-foundation/output"
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")

class Chunk(BaseModel):
    text: str
    chunk_id: str
    section_tag: Literal["References", "Content", "Appendix"]
    page_number: int = Field(ge=1)
    doc_id: str

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float

class Citation(BaseModel):
    page_number: int = Field(ge=1)
    doc_id: str

class AnswerSentence(BaseModel):
    text: str
    citations: list[Citation]

class Answer(BaseModel):
    sentences: list[AnswerSentence] = Field(min_length=1)

def is_appendix_heading(lines: list[str], i: int) -> bool:
    s = lines[i].strip()
    if s.lower() == "appendix":
        return True
    if s == "A":
        for nxt in lines[i + 1:]:
            nxt = nxt.strip()
            if nxt:                      # first non-empty line after the "A"
                return nxt[0].isupper()
    return False

def load_pages(path: str) -> list[str]:
    """Returns list of pages from the given file path.
    
    Args:
        path (str): The file path to each paper.
    
    The page number of item i is i + 1 (pages are numbered from 1, matching the PDF page index).
    Returns:
        list: A list of pages from the given file path.
    
    """
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return pages


def split_sections(pages: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Returns a dictionary containing section and section pages.
        
        Args:
            pages (list[str]): list of pages from the document.

        Definition:
            A section is defined as a part of the document that starts with a section heading and ends before the next section heading.
            If no heading, then the section is by default content for the text.
            If reference is in middle of page, text after the reference is considered part of the reference section.
        
        Returns:
            dict: A dictionary containing section and section pages, where the key is the section name and the value is a list of tuples containing the page number and the text on that page.
        
        """
    sections = {
        "Content": [],
        "References": [],
        "Appendix": []
    }
    ref_found = False
    appendix_found = False
    for page_num, page in enumerate(pages, start=1):
        ref_text = ""
        appendix_text = ""
        content = ""
        lines = page.split("\n")
        for i, line in enumerate(lines):
            if ref_found and not is_appendix_heading(lines, i):
                ref_text += line + "\n"
            elif appendix_found and line.strip().lower() != "references":
                appendix_text += line + "\n"
            elif line.strip().lower() == "references":
                if appendix_found:
                    appendix_found = False
                ref_found = True
                ref_text += line + "\n"
            elif is_appendix_heading(lines, i) and ref_found:
                if ref_found:
                    ref_found = False
                appendix_found = True
                appendix_text += line + "\n"
            else:
                content += line + "\n"
        if content != "":
            sections["Content"].append((page_num, content))
        if ref_text != "":
            sections["References"].append((page_num, ref_text))
        if appendix_text != "":
            sections["Appendix"].append((page_num, appendix_text))
    return sections

def clean_section(section: list[tuple[int, str]]) -> list[tuple[int, list[str]]]:
    """Cleans a section by removing any empty text entries/removing unwanted new line strings between sentences 
        or removing emtpy lines or removing unwanted data (for eg: lines with only digits or removing roman letters etc.).

        Args:
            section (list[tuple[int, str]]): A list of tuples containing the page number and the text on that page.

        Returns:
            list[tuple[int, list[str]]]: A cleaned list of tuples containing the page number and a list of paragraphs on that page.

    """

    cleaned_section = []
    roman_numeral = re.compile(
        r"(?i)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
    )
    for page_number, text in section:
        sentences = []
        page_text = ""
        text = text.split("arXiv Template\nA PREPRINT\n")[-1]
        lines = text.split("\n")
        non_empty = [i for i, l in enumerate(lines) if l.strip()]
        if non_empty:
            first, last = non_empty[0], non_empty[-1]
        else:
            first, last = 0, len(lines) - 1
        for i, line in enumerate(lines):
            line = line.strip()
            if i == first and (line.isdigit() or line.startswith("Published as a conference paper at")):
                continue
            if i == last and line.isdigit():
                continue
            page_text += line + "\n"
        paragraph = re.split(r"(?<=\.)\n", page_text)
        for sentence in paragraph:
            sentence = sentence.strip()
            sentence = sentence.replace("-\n", "")
            idx = sentence.find("\n")
            if idx != -1:
                sentence = sentence.replace("\n", " ")
            if (sentence.isdigit() or sentence.isspace() or not sentence.strip() or sentence.startswith(("arXiv","Preprint","URL","org","doi")) or ("arXiv" in sentence) or (roman_numeral.fullmatch(sentence)) or (sentence.isalpha() and len(sentence) ==1)) and (len(sentence) < 50):
                continue
            sentences.append(sentence)
        if sentences:
            cleaned_section.append((page_number, sentences))
    return cleaned_section


def chunk_paragraphs(section: list[tuple[int, list[str]]], section_tag: str, doc_id: str, chunk_size: int = 600, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunks a section paragraphs.

        Args:
            section (list[tuple[int, list[str]]]): A list of tuples containing the page number and a list of paragraphs on that page.
            section_tag (str): The tag or name of the section. Allowed values: ["References", "Content", "Appendix"]
            doc_id (str): The ID of the document.
            chunk_size (int, optional): The maximum size of each chunk in characters. Defaults to 600.
            chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 150.

        Paragraph upto chunk_size character is considered as a single chunk.
        If a paragraph short of chunk_size characters, it will be combined with the next paragraph to form a chunk of up to chunk_size characters, if the second paragraph can be added without breaking sentences.
        If a paragraph exceeds chunk_size characters, it will be split into multiple chunks, ensuring that each chunk ends at a sentence boundary and the succeding chunk starts with an overlap of chunk_overlap characters from preceeding chunk without breaking a sentence with a 10% character tolerance.
        if a chunk spans multiple pages, the page number will be recorded as the starting page.
        The chunking logic ensures that sentences are not broken and that overlaps are maintained for context continuity.
        Returns:
            list[Chunk]: A list of Chunk objects containing the chunked paragraphs, chunk id, section_tag, page number and doc_id.

    """
    flat = []
    for page_number, paragraphs in section:
        for paragraph in paragraphs:
            sentences = re.split(r"(?<!\b[A-Z]\.)(?<=[.!?])\s+(?=[A-Z])", paragraph)
            for sentence in sentences:
                if len(sentence) <= chunk_size:
                    flat.append((page_number, sentence))
                else:
                    words = sentence.split(" ")
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                flat.append((page_number, current_chunk.strip()))
                            current_chunk = word + " "
                    if current_chunk:
                        flat.append((page_number, current_chunk.strip()))
    buffer = []
    counter = 0
    chunks = []
    for page_number, sentence in flat:
        string = " ".join([s for _, s in buffer])
        if not buffer:
            size = len(string) + len(sentence)
        else:
            size = len(string) + len(sentence) + 1
        if size <= chunk_size:
            buffer.append((page_number, sentence))
        elif size > chunk_size and buffer:
            chunk_text = " ".join([s for _, s in buffer])
            chunk = Chunk(
                text=chunk_text,
                chunk_id=doc_id+f"_{section_tag}_{counter:04d}",
                section_tag=section_tag,
                page_number=buffer[0][0],
                doc_id=doc_id
            )
            pg_no, txt = buffer[-1]
            buffer = []
            if len(txt) <= chunk_overlap * 1.10 and len(sentence + txt) + 1 <= chunk_size:
                buffer.append((pg_no, txt))
            counter += 1
            buffer.append((page_number, sentence))
            chunks.append(chunk)
    if buffer:
        chunk = Chunk(
            text=" ".join([s for _, s in buffer]),
            chunk_id=doc_id+f"_{section_tag}_{counter:04d}",
            section_tag=section_tag,
            page_number=buffer[0][0],
            doc_id=doc_id
        )
        chunks.append(chunk)
    return chunks


def build_index(chunks: list[Chunk]) -> None:
    """Builds an index for the given list of chunks. Store vector embeddings for each chunk as numpy arrays and store the chunk text and metadatas as json file.
    The function expects the chunks from all 21 papers to be included in the list.

        Args:
            chunks (list[Chunk]): A list of Chunk objects to be indexed.
        Model used for embedding the chunk is nomic-ai/nomic-embed-text-v1.5
        The vector embeddings row should match the order of metadata entries.
        The numpy arrays and metadata json will be stored as separate files under rag-foundation/output.
        Returns:
            None

    """
    metadata_json = []
    embedding_texts = []
    for chunk in chunks:
        chunk_text = chunk.text
        embedding_text = "search_document: " + chunk_text
        embedding_texts.append(embedding_text)
        metadata_json.append({
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "section_tag": chunk.section_tag
        })
    embeddings = model.encode(embedding_texts, 
                        normalize_embeddings=True, 
                        batch_size=16,
                        show_progress_bar=True)
    print(f"Number of embedding texts: {len(embedding_texts)}")
    print(f"Shape of embeddings: {embeddings.shape}")
    assert len(embedding_texts) == embeddings.shape[0]
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    np.save(f"{OUTPUT_PATH}/index.npy", embeddings)
    with open(f"{OUTPUT_PATH}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f)

def load_index(path: str) -> tuple[np.ndarray, list[Chunk]]:
    """Loads the index and metadata from the given path.

        Args:
            path (str): The path to the index files - rag-foundation/output.

        The path corresponds to the directory where the index files are stored should be same as during the build_index function.
        The index files are expected to be in the format of numpy arrays for vector embeddings and JSON files for metadata.
        Returns:
            tuple[np.ndarray, list[Chunk]]: A tuple containing the vector embeddings as a numpy array and the metadata as a list of chunks.

    """
    index = np.load(f"{path}/index.npy")
    with open(f"{path}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = [Chunk(**chunk_data) for chunk_data in metadata]
    assert len(index) == len(chunks)
    return index, chunks

def retrieve(query: str, index: np.ndarray, chunks: list[Chunk], top_k: int = 5, section_tag: str = "Content") -> list[RetrievedChunk]:
    """Retrieves the top_k most similar chunks to the query using the index.

        Args:
            query (str): The query string to search for.
            index (np.ndarray): The numpy array containing vector embeddings.
            chunks (list[Chunk]): The list of chunks corresponding to the index.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            section_tag (str, optional): The section tag to consider for similarity. Defaults to "Content".
        Use the embedding model loaded nomic-ai/nomic-embed-text-v1.5 to embed the query.
        Vectors are normalized before computing similarity.
        User query need to be changed as search_query before passing to model
        Sort the chunks based on their similarity score to the query embedding.
        Returns:
            list[RetrievedChunk]: A list of RetrievedChunk objects representing the top_k most similar chunks.

    """
    query = f"search_query: {query}"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    idx = [i for i, chunk in enumerate(chunks) if chunk.section_tag == section_tag]
    if not idx:
        raise ValueError(f"No chunks found with section_tag: {section_tag}")
    chunks_filtered = [chunks[i] for i in idx]
    index_filtered = index[idx]
    similarities = index_filtered @ query_embedding
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_chunks = []
    for pos in top_indices:
        chunk = chunks_filtered[pos]
        retrieved_chunks.append(RetrievedChunk(chunk=chunk, score=np.float32(similarities[pos])))
    return retrieved_chunks

def check_citation(answer: Answer, retrieved_chunks: list[RetrievedChunk]) -> list[tuple[bool, Citation]]:
    """Checks the citation in the given answer.

        Args:
            answer (Answer): The answer object containing the answer sentences and citations.
            retrieved_chunks (list[RetrievedChunk]): The list of retrieved chunks relevant to the answer.
        Validates the citations to ensure they correctly reference the source chunks used in generating the answer.
        If any citations are incorrect or missing, they should be flagged for correction.
        Citation should contain page number and doc_id as per the retrieved chunks.
        A valid citation means that the page number and doc_id correctly correspond to the source chunk used in generating the answer.
        Returns:
            list[tuple[bool, Citation]]: A list of tuples containing a boolean indicating the validity of the citation and the corresponding citation object.

    """
    results = []
    for sentence in answer.sentences:
        for citation in sentence.citations:
            is_valid = False
            for chunk in retrieved_chunks:
                if chunk.chunk.doc_id == citation.doc_id and chunk.chunk.page_number == citation.page_number:
                    is_valid = True
                    break
            results.append((is_valid, citation))
    return results

def answer(query: str, retrieved_chunks: list[RetrievedChunk]) -> Answer:
    """Generates an answer sentence based on the search query and retrieved chunks.

        Args:
            query (str): The search query string.
            retrieved_chunks (list[RetrievedChunk]): The list of retrieved chunks relevant to the search query.

        Ollama model 'llama3.2:3b' is used to generate the answer based on the retrieved chunks and the query.
        The answer is generated based on the content of the retrieved chunks and the query.
        Citations are included to reference the source chunks used in generating the answer.
        If the retrieved chunks do not contain sufficient information to answer the query, respond  saying Sorry context doesn't have enough details to answer the question.
        Validate the model output to ensure it returns the expected outcome as per the Answer model.
        Returns:
            Answer: A list of generated answer sentences and corresponding citations.

    """
    prompt = f"You are an AI assistant. Answer the following user query based on the provided context."
    rule = "\nRules:\n - Only use the information provided in the context to answer the query.\n - Include citations to the relevant chunks.\n - If the information is insufficient, respond with the refusal message. \n - Copy doc_id and page exactly as in the chunk headers. \n - Always return at least one sentence; a refusal is one sentence with an empty citations list. \n - Respond in JSON with a sentences list, where each item has text and a citations list of doc_id and page_number. \n - Answer only if the chunk/context directly answers the user query, never try to link an irrelevant content with user query."
    prompt += rule
    context_prompt = "\nContext:"
    prompt += context_prompt
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"\nChunk Text: ({chunk.chunk.doc_id},{chunk.chunk.page_number}) {chunk.chunk.text}"
    refusal_prompt = "\nIf the retrieved chunks do not contain sufficient information to answer the query, respond with 'Sorry context doesn't have enough details to answer the question.'"
    prompt += refusal_prompt
    output_prompt = "\nPlease provide the answer with citations to the relevant chunks, citations should include doc_id and page_number."
    prompt += output_prompt
    prompt += f"\nQuery: {query}"
    response = chat(
        model='llama3.2:3b',
        format=Answer.model_json_schema(),
        options={'temperature': 0.0},
        messages=[{'role': 'user', 'content': prompt}],
    )
    answer_text = response.message.content
    return Answer.model_validate_json(answer_text)
