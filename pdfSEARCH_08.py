import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import pickle
import os
from textblob import TextBlob
import markdown
import re
import nltk
from nltk.corpus import stopwords

# Download the stopwords data
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarkdownSearchConfig:
    def __init__(self, topk_results=16):
        self.TOPK_RESULTS = topk_results

def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def split_markdown_into_chunks(md_text, source_file):
    md = markdown.Markdown(extensions=['meta'])
    html = md.convert(md_text)

    # Extract page numbers from the Markdown
    pages = [p.strip() for p in md_text.split('Page ') if p.strip()]

    chunks = []
    line_number = 1
    char_position = 0
    page_images = {}

    for page_number, page in enumerate(pages, start=1):
        # Extract image tags and descriptions within the page
        image_filenames = []
        image_descriptions = []

        for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', page):
            description = match.group(1)
            filename = match.group(2)
            image_filenames.append(filename)
            image_descriptions.append(description)

        # Remove the image tags from the page content
        page_without_images = re.sub(r'!\[.*?\]\(.*?\)', '', page)

        # Store images for the page
        page_images[page_number] = {
            'image_filenames': image_filenames,
            'image_descriptions': image_descriptions
        }

        # Split the page into paragraphs
        paragraphs = [p.strip() for p in page_without_images.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            # Split the paragraph into sentences
            sentences = [s.strip() for s in paragraph.split('. ') if s.strip()]
            for sentence in sentences:
                start_char = char_position
                end_char = char_position + len(sentence)

                chunk = {
                    'text': clean_text(sentence),
                    'source': source_file,
                    'page_number': page_number,
                    'start_char': start_char,
                    'end_char': end_char,
                    'line_number': line_number
                }

                chunks.append(chunk)

                char_position = end_char + 1
                line_number += 1

    return chunks, page_images

def calculate_keyword_frequency(chunks):
    keyword_frequency = {}
    for chunk in chunks:
        words = chunk['text'].lower().split()
        for word in words:
            if word in keyword_frequency:
                keyword_frequency[word] += 1
            else:
                keyword_frequency[word] = 1
    return keyword_frequency

def searchFLOWpdf2md_keyword(file_path, query, config):
    if not query:
        raise ValueError("The query text must not be empty or None")

    cache_path = f"{file_path}_markdown.pickle"
    if os.path.exists(cache_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            data = pickle.load(cache_file)
    else:
        logging.info("Processing data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read().strip()
        chunks, page_images = split_markdown_into_chunks(md_text, file_path)
        keyword_frequency = calculate_keyword_frequency(chunks)
        data = {
            'chunks': chunks,
            'page_images': page_images,
            'keyword_frequency': keyword_frequency
        }
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logging.info(f"Data processed and cached: {cache_path}")

    chunks = data['chunks']
    page_images = data['page_images']
    keyword_frequency = data['keyword_frequency']

    # Perform keyword search
    stop_words = set(stopwords.words('english'))
    query_keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]
    relevant_chunks = []

    for chunk in chunks:
        chunk_keywords = chunk['text'].lower().split()
        if any(keyword in chunk_keywords for keyword in query_keywords):
            relevant_chunks.append(chunk)

    # Append image descriptions to relevant chunks if on the same page
    for chunk in relevant_chunks:
        page_info = page_images.get(chunk['page_number'], {})
        if page_info:
            image_descriptions = page_info.get('image_descriptions', [])
            if image_descriptions:
                chunk['text'] += " " + " ".join(f"{desc}" for desc in image_descriptions)
                chunk['image_filenames'] = page_info.get('image_filenames', [])
                chunk['image_descriptions'] = image_descriptions

    # Sort relevant chunks based on keyword frequency
    relevant_chunks.sort(key=lambda x: sum(keyword_frequency.get(word, 0) for word in x['text'].lower().split()), reverse=True)

    # Clean the chunks by removing (filename) and other broken un escaped characters
    for chunk in relevant_chunks:
        chunk['text'] = re.sub(r'\(.*?\)', '', chunk['text'])
        chunk['text'] = re.sub(r'[^a-zA-Z0-9\s]', '', chunk['text'])

    # Get top-k results
    top_results = relevant_chunks[:config.TOPK_RESULTS]

    if not top_results:
        # Return an empty list if no relevant chunks are found
        return []

    # Calculate relevance scores based on keyword frequency and position
    max_keyword_freq = max(sum(keyword_frequency.get(word, 0) for word in chunk['text'].lower().split()) for chunk in top_results)
    for i, chunk in enumerate(top_results):
        keyword_freq = sum(keyword_frequency.get(word, 0) for word in chunk['text'].lower().split())
        relevance_score = keyword_freq / max_keyword_freq
        position_score = 1 - (i / len(top_results))
        chunk['relevance_score'] = round(relevance_score * position_score, 3)

    results_with_sentiment = []
    for chunk in top_results:
        sentiment = round(TextBlob(chunk['text']).sentiment.polarity, 3)
        relevance_score = chunk['relevance_score']
        results_with_sentiment.append((chunk, sentiment, relevance_score))

    return results_with_sentiment

if __name__ == '__main__':
    config = MarkdownSearchConfig(topk_results=8)
    file_path = "WALTER_RUSSEL/WalterRusselbooks/Walter Russell - books/Home Study Course/Home Study Course Unit 2 Lessons 5,6,7,8 by Walter Russell/Home Study Course Unit 2 Lessons 5,6,7,8 by Walter Russell.md"
    query = 'handwritten page'
    results = searchFLOWpdf2md_keyword(file_path, query, config)
    for chunk, sentiment, relevance_score in results:
        print(f"Chunk: {chunk['text']}")
        print("---")
        source = chunk.get('source', 'No source available')
        print(f"Source: {source}")
        print(f"Image Descriptions: {chunk.get('image_descriptions', [])}")
        print(f"Image Filenames: {chunk.get('image_filenames', [])}")
        print(f"Page Number: {chunk.get('page_number', 'N/A')}")
        print(f"Start Char: {chunk.get('start_char', 'N/A')}")
        print(f"End Char: {chunk.get('end_char', 'N/A')}")
        print(f"Line Number: {chunk.get('line_number', 'N/A')}")
        print(f"Relevance Score: {relevance_score:.3f}")
        print(f"Sentiment: {sentiment:.3f}")
        print("---")