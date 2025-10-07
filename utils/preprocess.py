import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

def summarize_document(doc_text):
    """
    Generates a compact semantic summary of the document for high-level embedding
    using the Databricks-hosted LLM. The summary is later used for document-level retrieval.
    """
    if not doc_text.strip():
        return "Empty document."

    prompt = f"""
Summarize the main topics purpose of the following documentation page in 1-2 sentences to be used in retrieval in a rag pipeline."""

    try:
        response = openai_client.chat.completions.create(
            model="databricks-gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a Databricks documentation assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )
        return response.choices[0].message.content[1]["text"]
    except Exception as e:
        print(f"⚠️ Summarization failed for {e}")
        return "Summary unavailable due to error."

# -------------------------------------------------------------------------
# 🔹 Function: Section-aware Chunking
# -------------------------------------------------------------------------
def perform_section_chunking(document, url, doc_id, chunk_size=1200, chunk_overlap=200):
    """
    Performs section-aware chunking for structured docs like Databricks documentation.
    No language model used — purely regex + text splitter.
    """
    text = re.sub(r"â€™", "'", document)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    section_pattern = re.compile(
        r"(?<=\n)([A-Z][A-Za-z0-9\s\-]+(?:for the workspace|management|tokens?|users?|principals?)?)\n\1\n"
    )

    sections = []
    last_end = 0
    for match in section_pattern.finditer(text):
        header = match.group(1).strip()
        start = match.start()
        if last_end < start:
            sections.append((header, text[last_end:start].strip()))
        last_end = match.end()

    if last_end < len(text):
        sections.append(("Miscellaneous", text[last_end:].strip()))

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    documents = []
    global_counter = 0  # <── add this

    for section_title, section_text in sections:
        chunks = splitter.split_text(section_text)
        for chunk in chunks:
            documents.append({
                "doc_id": doc_id,
                "url": url,
                "section": section_title,
                "chunk_id": f"{doc_id}_{global_counter}",  # use global counter
                "chunk_text": chunk,
                "chunk_size": len(chunk)
            })
            global_counter += 1  # increment globally
    return documents
