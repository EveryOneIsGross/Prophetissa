# v 28
import json
import argparse
import re
from openai import OpenAI
from feelSEARCH import searchFLOWfeels, feelsConfig
from hybrid25SEARCH import searchFLOWhybrid, hybridConfig
from pdfSEARCH import searchFLOWpdf2md_keyword, MarkdownSearchConfig
from wikiSEARCH import searchFLOWwiki, WikiSearchConfig
from fractalSEARCH import searchFLOWfractals, fractalSearchConfig

TOPK = 4
MAXTOKENS = 1028
CHUNK_SIZE = 256
N_CHUNKS = 1

QUESTIONMODEL = "qwen2:0.5b-instruct-fp16"
ANSWERMODEL = "qwen2:0.5b-instruct-fp16"

seen_entries = set()

# Initialize the OpenAI client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

def insert_citations(text_segments, citations):
    """
    Insert citations into the text after each chunk and append full citation details at the end.
    """
    formatted_text = ""
    full_citations = []

    for i, segment in enumerate(text_segments):
        citation = citations[i]
        page_number = citation.get('page_number', 'N/A')
        citation_marker = f"[{i+1}, {page_number}]"
        quoted_text = f'"{segment}"{citation_marker}'

        formatted_text += quoted_text
        if i < len(text_segments) - 1:
            formatted_text += "\n\n"  # Add separation between chunks

        full_citation = f"[{i+1}]: Source: {citation['source']}, Start Char: {citation['start_char']}, End Char: {citation['end_char']}, Page: {page_number}"
        full_citations.append(full_citation)

    full_citations_text = "\n".join(full_citations)
    formatted_text += "\n\n" + full_citations_text
    return formatted_text

def generate_questions(context, num_questions):
    prompt = f"""
Context information is below.
Analyze the text and identify the following:
- Key topics, entities, events, and relationships
- Temporal and spatial information
- Quantitative data and comparisons
- Potential areas for further exploration or questioning
- Evaluate the content and choose the appropriate type of questioning.

Simple - Questions asking for simple facts that are unlikely to change over time.
Conditional - Questions asking for facts with some given conditions or constraints.
Set - Questions that expect a set of entities or objects as the answer.
Comparative - Questions that compare two or more entities, events, or concepts.
Aggregative - Questions that require aggregation of information to answer.
Inferential - Questions that require making inferences based on the given information.
Analytical - Questions needing reasoning or analysis of the passage to obtain the answer.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, generate {num_questions}
questions based on the context. The questions should be diverse in nature across the
document. 

Restrict the questions to the context information provided. Each question should be a single sentence ending with a question mark or a newline character.
"""
    response = client.chat.completions.create(
        model=QUESTIONMODEL,
        messages=[
            {"role": "system", "content": """You are a curious agent."""},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response.choices[0].message.content.strip()
    generated_questions = [q.strip() for q in generated_text.split('\n') if q.strip().endswith('?')]
    generated_questions = [re.sub(r'^\d+\)', '', q) for q in generated_questions]
    generated_questions = [re.sub(r'^\d+\.', '', q) for q in generated_questions]
    generated_questions = [q[q.find(next((char for char in q if char.isupper()), q[0])):] for q in generated_questions]
    generated_questions = generated_questions[:num_questions]
     
    print(prompt)
    print(generated_questions)
    return generated_questions

def generate_answer(context, question):
    prompt = f"""
Context information is below
---------------------
{context}
---------------------
**Instruction:**

Given the above format for each chunk, follow these steps to create the TLDR; summary:

1. **Review All Chunks**: Go through each provided chunk to understand the content, paying attention to the relevance and sentiment scores to gauge the importance and tone.
2. **Identify Key Points**: Extract the most important points from the chunks, prioritizing those with higher relevance scores.
3. **Condense Information**: Summarize the key points in a concise manner, ensuring that the essence of the information is retained.
4. **Maintain Clarity and Coherence**: Ensure that the summary is clear, coherent, and makes sense as a standalone text without requiring the reader to refer back to the chunks.
5. **Use Neutral Language**: Maintain a neutral tone unless the sentiment score indicates a strong positive or negative tone, in which case, reflect this sentiment appropriately.

**Generated TLDR; Summary:**

"Artificial intelligence is significantly enhancing various industries through automation, efficiency improvements, and data-driven insights. However, the adoption of AI faces challenges, including data privacy issues, ethical concerns, and a shortage of specialized skills."

---

Ensure to follow this format and instruction for each set of provided chunks to generate effective and accurate TLDR; summaries.
Given the context information and not prior knowledge,
answer the query.

Query: 
{question}
Answer: """
    response = client.chat.completions.create(
        model=ANSWERMODEL,
        messages=[
            {"role": "system", "content": """You are an advanced summarization AI capable of generating concise TLDR; summaries for given text chunks. Each chunk will be provided with the text, start and end character positions, relevance score, sentiment score, and source file information. Your task is to read through the provided chunks and create a summary that captures the key points succinctly while retaining the most relevant and informative content. Here is the format for each chunk that will be provided:"""},
            {"role": "user", "content": prompt}
        ]
    )
    generated_answer = response.choices[0].message.content.strip()
    print(prompt)
    print(generated_answer)
    return generated_answer

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read().strip()
            if data:
                return json.loads(data)
            else:
                return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

import numpy as np

def save_json_data(file_path, data):
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2, default=convert_to_serializable)

def process_search_results(file_path, query, output_file, search_tool):
    if search_tool == 'hybrid':
        searchCONFIG = hybridConfig(topk_results=TOPK, chunk_size=CHUNK_SIZE, threshold=20, k1=1.5, b=0.75)
        search_results = searchFLOWhybrid(file_path, query, searchCONFIG)
    elif search_tool == 'feels':
        searchCONFIG = feelsConfig(topk_results=TOPK, chunk_size=CHUNK_SIZE, max_tokens=MAXTOKENS)
        search_results = searchFLOWfeels(file_path, query, searchCONFIG)
    elif search_tool == 'fractal':
        searchCONFIG = fractalSearchConfig(topk_results=TOPK, initial_chunk_size=CHUNK_SIZE, max_iter=3, min_chunk_size=1)
        search_results, _ = searchFLOWfractals(query, file_path, searchCONFIG)
    elif search_tool == 'pdf2md':
        searchCONFIG = MarkdownSearchConfig(topk_results=TOPK)
        search_results = searchFLOWpdf2md_keyword(file_path, query, searchCONFIG)
    elif search_tool == 'wikisearch':
        searchCONFIG = WikiSearchConfig(n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE, max_tokens=MAXTOKENS, topk_results=TOPK)
        wiki_search_results = searchFLOWwiki(query, searchCONFIG)

        if wiki_search_results is None:
            print(f"No search results found for query: {query}")
            return
        
        search_results = []
        for wiki_data in wiki_search_results:
            search_results.extend(wiki_data['search_results'])
    
    existing_data = load_json_data(output_file)
    text_segments = []
    citations = []
    chunks_data = []

    is_pdfmd = search_tool == 'pdf2md'
    is_wikisearch = search_tool == 'wikisearch'
    is_fractal = search_tool == 'fractal'

    for i, result in enumerate(search_results):
        if is_pdfmd:
            chunk, sentiment, relevance_score = result
            start_char = chunk.get('start_char', 'N/A')
            end_char = chunk.get('end_char', 'N/A')
            page_number = chunk.get('page_number', 'N/A')
            line_number = chunk.get('line_number', 'N/A')
            image_filenames = chunk.get('image_filenames', [])
            image_descriptions = chunk.get('image_descriptions', [])
            source = chunk['source']
            if image_filenames:
                source += f" (Images: {', '.join(image_filenames)})"
            text_segments.append(chunk['text'])
        elif is_wikisearch:
            chunk = result['text']
            sentiment = wiki_search_results[i // N_CHUNKS]['sentiment_polarity']
            relevance_score = wiki_search_results[i // N_CHUNKS]['keyword_frequency']
            start_char = result['start_char']
            end_char = result['end_char']
            page_number = "N/A"
            line_number = "N/A"
            image_filenames = []
            image_descriptions = []
            source = wiki_search_results[i // N_CHUNKS]['url']
            text_segments.append(chunk)
        elif is_fractal:
            file_path, chunk_text, start_char, end_char, sentiment, relevance_score = result
            source = file_path
            page_number = "N/A"
            line_number = "N/A"
            image_filenames = []
            image_descriptions = []
            text_segments.append(chunk_text)
        else:
            chunk, sentiment = result[:2]
            relevance_score = result[2] if not is_pdfmd else None
            start_char = chunk.get('start_char', 'N/A')
            end_char = chunk.get('end_char', 'N/A')
            page_number = "N/A"
            line_number = "N/A"
            image_filenames = []
            image_descriptions = []
            source = chunk['source']
            text_segments.append(chunk['text'])

        citations.append({
            "start_char": start_char,
            "end_char": end_char,
            "source": source,
            "page_number": page_number,
            "chunk_number": i + 1,
            "line_number": line_number
        })

        chunks_data.append({
            "text": chunk_text if is_fractal else (chunk if is_wikisearch else chunk['text']),
            "start_char": start_char,
            "end_char": end_char,
            "page_number": page_number,
            "line_number": line_number,
            "image_filenames": image_filenames,
            "image_descriptions": image_descriptions,
            "sentiment": sentiment,
            "relevance_score": relevance_score
        })

    formatted_context = insert_citations(text_segments, citations)

    num_questions = 3
    questions = generate_questions(formatted_context, num_questions)

    for question in questions:
        answer = generate_answer(formatted_context, question)
        entry_key = (query, question, answer)

        if entry_key in seen_entries:
            print(f"Duplicate entry found for query: {query}, question: {question}. Skipping.")
            continue

        seen_entries.add(entry_key)

        print(formatted_context)
        print("---")

        result = {
            "query": query,
            "context": formatted_context,
            "citations": citations,
            "chunks": chunks_data,
            "question": question,
            "answer": answer
        }
        existing_data.append(result)
        save_json_data(output_file, existing_data)
        print(f"Search results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some search results.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--seed_queries_file', type=str, required=True, help='Path to the seed queries file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--search_tool', type=str, default='hybrid', choices=['hybrid', 'feels', 'pdf2md', 'wikisearch', 'fractal'], help='Search tool to use: "hybrid", "feels", "pdf2md", "wikisearch", or "fractal"')
    
    args = parser.parse_args()
    with open(args.seed_queries_file, 'r') as file:
        seed_queries = file.read().splitlines()

    for query in seed_queries:
        if not query:
            print("Received an empty query from seed_queries_file, skipping...")
            continue
        print(f"Processing query: {query}")
        process_search_results(args.file_path, query, args.output_file, args.search_tool)
