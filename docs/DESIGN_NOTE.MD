# **RAG Baseline Pipeline: Design Note (Challenge 1)**
Baseline RAG over 21 arXiv papers: PyMuPDF, recursive paragraph/sentence chunking (600/150), nomic-embed-text-v1.5, numpy plus JSON index, cited answers.

## **Indexing (Offline)**
### Read the document
Input: Read documents from /Users/tomthomas/Documents/rag-foundation/data/papers/. Each file name (arXiv id) will serve as document id in metadata

Output: List of pages for each paper, where page number is identified as page.number + 1 and it Is used for final LLM generated output as citation.

Decision: PyMuPDF library is used for reading the document

Reason: After testing different libraries like pypdf - no paragraph structure, pdfplumber - no space between words, everything is stitched together, PyMuPDF outperformed based on the output it provided with  spaces between words.

### **Separate out references and appendix with separate tags from actual content**

Input: List of pages per paper read using PyMuPDF library

Output: Separate text for content, reference and appendix

Decision: Start with default false for ref_found. For every line in each page of the paper, check for the line that only has references in lower case, attach the text to separate ref_text string and maintain its page number for future citation, make ref_found to true and continue attaching the following strings after ref_found to same ref_text string, so we have content and reference separated. Similarly do it for appendix if its present

Reason: If we simply look  for finding “Reference” in document to separate out references, we may end up finding wrong scenarios, like the reference mentioned in between sentences. If we particularly look for”./nReferecne” we will end up depending on the previous sentence to have a full stop to find reference and its not necessary that previous sentence will have full stop. So the best approach was to read through each line and find out where the line only has reference in it
Tested only on one paper, yet to be tested on all 21 papers together.

### **Clean the document**
Input: List of pages for each papers

Output: List of Paragraph seperated cleaned document. Cleaned document with \n between lines cleared, sentence break with hypen addressed, arxiv template stripped of from document, lines with single numerals or letters or empty spaces are removed

Decision: Use python code to clean the data. First the text in pages are broken down to paragraphs using “./n” separator, then /n between the sentences are figured out and removed using find operation, then finally hypens seperated words are stitched together. This processed text is passed for further cleaning like removal of roman letters, lines with single characters or digits, lines with empty spaces.

Reason: This is done to remove unwanted data from getting embedded there by misguiding the retrieval

### **Chunking (Manual)**
Input: List of cleaned paragraphs

Output: Chunks of data with 600 chunk size and an overlap of 150

Decision: For every document, the size of each paragraph has been checked, and it was observed that with a chunk size of 600, it would cover 80 to 90 percentage of paragraph as a chunk, so breaking a paragraph into two is avoided. Also looking at the length of sentences, it was observed that 75 to 85% of sentences have 150 characters, so we are just bringing in only 1 sentence max as overlap not more than that.

Now for chunking, every paragraph from list of paragraph is taken its length is compared to threshold of 600, if its 600 its considered as 1 chunk, if its more than 600 it will be broken down to two chunks in such a way that first chunk is closer to 600 chunks and the breaking happens exactly at a sentence break, the first chunk contains upto 600 characters, the second chunk contains a complete sentence which can go 150 or more with some tolerance to make sure larger sentences are not broken down during overlap characters from first chunk and the next coming paragraph to complete upto 600 chars for second chunk, and it will go like that. And if first paragraph doesn’t have 600 chars, then the second one will also be used to complete the required 600 chars if second paragraph fits completely to first paragraph to make it 600 chars, keep in mind 600 is ceiling doesn’t mean chunk size cannot go less that 600. When breaking down to chunks, it will be made sure the text under same heading will be part of same chunk, for eg intorduction and methodology wont mix together. Also if a paragraph is at page break, it will be stictched with the content in next page

Reason: The chunking is done in proper analyzed way so that we will end up having an efficient retrieval

### **Embedding and storage**
Input: Chunks of data along with page num, chunk id, doc_id (document id of arxiv), reference tag (reference/content/appendix)

Output: vector embedding and a searchable vector db

Decision: nomic-ai/nomic-embed-text-v1.5 this is the model going to be used for embedding with 8K token limit. The chuks of data is passed to the model for embedding. The embedded data is stored as numpy array with text corresponding to the embedding and the corresponding metadatas like page number, chunk_id, reference_tag and doc_id are stored as separate json. Then the embedded user query is used to search and retrieve content using cosine similarity. when embedding the document, we use search_document as prefix and during retrieval we use search_query

Reason: Embedded data stored in numpy array  along with metadata as separate json will help in effficient retrieval and the metadata along with retrieved chunk help in citation Metadata like section tag by default set for content always, can be filtered for reference or appendix as required. The embedding and metadata generation happens only ones (indexing) which is then reused for retrieval. The embeded data stored as numpy array and metadata in json will be stored locally for resue.

## **Query phase (Online)**
Input: User question/query

Output: Answer to user query with citation pointing to the reference document
Decision: The user query is prepend with search_query and embedded using the same model used to embed the seach_document or reference documents. The normalised embeddings are used to apply cosine similarity, based on the pre-deifned top_k (k=3), number of top matching documents are retirevd documents are retrieved. The retrieved document matching to the top embedding is fed into the llm along with metadata like page number, doc_id.  The output will be pydantic model with answers separated as sentences with each sentence having corresponding citations.

```Sample LLM prompt: Answer the user question {question} using the retrieved context, when answering the question follow the exact format in the example to give citation for the answer. Use only details from retrieved context for citation and answering user question. If the context doesn’t have enough details corresponding to user query, just respond “Sorry the context doesn’t have enough details to answer the question”```
context: {retrieved document, metadata}
Example: Question: What is RAG?
Answer: RAG stands for Retrieval Augmented Generation[Doc_id: 2145:2330, page no:2]
 Before responding to user query make sure the citation in the llm generated output is part of retrieved chunk metadata

Reason: The answer generated with user context and citation give proper clarity and explanation to the user
