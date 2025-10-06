from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def perform_semantic_chunking(document, url, doc_id, chunk_size=1024, chunk_overlap=128):
    """
    Performs semantic chunking on a document using recursive character splitting
    and adds metadata for section info and density.
    """
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Clean the document text
    text = document.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"â€™", "'", text)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Split into chunks
    semantic_chunks = text_splitter.split_text(text)

    section_patterns = [
        r'^#+\s+(.+)$',      # Markdown headers
        r'^.+\n[=\-]{2,}$',  # Underlined headers
        r'^[A-Z\s]+:$',      # ALL CAPS section titles
        r'^(Step\s+\d+:)'    # Step headers
    ]

    documents = []
    current_section = url.split("/")[-1].replace("-", " ").title()

    for i, chunk in enumerate(semantic_chunks):
        chunk_lines = chunk.split('\n')
        for line in chunk_lines:
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    current_section = match.group(0)
                    break

        # Calculate semantic density
        words = re.findall(r'\b\w+\b', chunk.lower())
        stopwords = {'the','and','is','of','to','a','in','that','it','with','as','for'}
        content_words = [w for w in words if w not in stopwords]
        semantic_density = len(content_words) / max(1, len(words))

        documents.append({
            "doc_id": doc_id,
            "url": url,
            "chunk_id": i,
            "chunk_text": chunk,
            "chunk_size": len(chunk),
            "section": current_section,
            "semantic_density": round(semantic_density, 3)
        })

    return documents