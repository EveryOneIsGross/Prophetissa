import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import logging
import pickle
import os
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import math
from gpt4all import Embed4All
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class hybridConfig:
    def __init__(self, topk_results=16, chunk_size=32, threshold=20, k1=1.5, b=0.75): 
        self.TOPK_RESULTS = topk_results
        self.CHUNK_SIZE = chunk_size
        self.THRESHOLD = threshold
        self.K1 = k1
        self.B = b

def split_text_into_sentences(text, source_file):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    delimiters = {'.', '\n'}
    
    sentences = []
    current_sentence = []
    current_length = 0
    text_offset = 0

    for token_id in tokens:
        token = encoding.decode([token_id])
        current_sentence.append(token_id)
        current_length += 1

        if token in delimiters:
            sentence_text = encoding.decode(current_sentence)
            start_char = text_offset
            end_char = start_char + len(sentence_text) - 1
            sentences.append({
                'text': sentence_text,
                'start_char': start_char,
                'end_char': end_char,
                'source': source_file
            })
            current_sentence = []
            current_length = 0
            text_offset = end_char + 1

    if current_sentence:
        sentence_text = encoding.decode(current_sentence)
        start_char = text_offset
        end_char = start_char + len(sentence_text) - 1
        sentences.append({
            'text': sentence_text,
            'start_char': start_char,
            'end_char': end_char,
            'source': source_file
        })
    
    return sentences

def dynamic_chunk(sentences, embeddings, chunk_size, threshold):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    chunk_embeddings = []
    current_chunk = []
    current_embedding_chunk = []
    current_length = 0

    for sentence, embedding in zip(sentences, embeddings):
        sentence_tokens = encoding.encode(sentence['text'])
        if current_length + len(sentence_tokens) <= chunk_size + threshold:
            current_chunk.append(sentence)
            current_embedding_chunk.append(embedding)
            current_length += len(sentence_tokens)
        else:
            chunk_text = ''.join([s['text'] for s in current_chunk])
            start_char = current_chunk[0]['start_char']
            end_char = current_chunk[-1]['end_char']
            chunk_embedding = np.mean(current_embedding_chunk, axis=0)
            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'source': current_chunk[0]['source']
            })
            chunk_embeddings.append(chunk_embedding)
            current_chunk = [sentence]
            current_embedding_chunk = [embedding]
            current_length = len(sentence_tokens)
    
    if current_chunk:
        chunk_text = ''.join([s['text'] for s in current_chunk])
        start_char = current_chunk[0]['start_char']
        end_char = current_chunk[-1]['end_char']
        chunk_embedding = np.mean(current_embedding_chunk, axis=0)
        chunks.append({
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'source': current_chunk[0]['source']
        })
        chunk_embeddings.append(chunk_embedding)

    return chunks, chunk_embeddings

def compute_bm25(tf, df, N, dl, avdl, k1, b):
    if (N - df + 0.5) / (df + 0.5) + 1 <= 0:
        return 0
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    return idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * (dl / avdl))))

def semantic_search_with_bm25(embeddings, query_embedding, chunks, top_k, query, config):
    documents = [chunk['text'] for chunk in chunks]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    total_docs = len(documents)
    doc_lengths = X.sum(axis=1)
    avg_dl = doc_lengths.mean()

    query_vec = vectorizer.transform([query]).toarray()
    bm25_scores = np.zeros(X.shape[0])

    for i, j in zip(*X.nonzero()):
        bm25_scores[i] += compute_bm25(X[i, j], np.log((total_docs - np.sum(X[:, j] > 0) + 0.5) / (np.sum(X[:, j] > 0) + 0.5)), total_docs, doc_lengths[i, 0], avg_dl, config.K1, config.B)
    
    # Normalize BM25 scores using min-max scaling
    scaler = MinMaxScaler()
    bm25_scores = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    
    cosine_scores = cosine_similarity([query_embedding], embeddings).flatten()
    
    # Normalize cosine scores using L2 normalization
    cosine_scores = cosine_scores / np.linalg.norm(cosine_scores)
    
    final_scores = bm25_scores * cosine_scores
    rounded_scores = np.round(final_scores, 3)

    top_indices = np.argsort(rounded_scores)[-top_k:][::-1]
    return [(chunks[i], rounded_scores[i]) for i in top_indices]

def load_or_process_data(file_path, embedder):
    cache_path = f"{file_path}_hybrid.pickle"
    if os.path.exists(cache_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    else:
        logging.info("Processing data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        sentences = split_text_into_sentences(text, file_path)
        embeddings = [embedder.embed(sentence['text']) for sentence in sentences]
        data = {
            'sentences': sentences,
            'embeddings': embeddings
        }
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logging.info(f"Data processed and cached: {cache_path}")
        return data

def searchFLOWhybrid(file_path, query, config):
    if not query:
        raise ValueError("The query text must not be empty or None")

    embedder = Embed4All('nomic-embed-text-v1.f16.gguf')
    data = load_or_process_data(file_path, embedder)
    query_embedding = embedder.embed(query)
    sentences = data['sentences']
    embeddings = data['embeddings']

    # Create chunks based on the specified chunk size
    chunks, chunk_embeddings = dynamic_chunk(sentences, embeddings, config.CHUNK_SIZE, config.THRESHOLD)

    search_results = semantic_search_with_bm25(chunk_embeddings, query_embedding, chunks, config.TOPK_RESULTS, query, config)
    
    results_with_sentiment = []
    for chunk, score in search_results:
        sentiment = round(TextBlob(chunk['text']).sentiment.polarity, 3) # round
        results_with_sentiment.append((chunk, score, sentiment))
    
    return results_with_sentiment

if __name__ == '__main__':
    config = hybridConfig(topk_results=1, chunk_size=512, k1=1.5, b=0.75)
    file_path = 'conTEXTS\\text\\20th Century Women - The Script Lab1716430313.792667.txt'
    query = 'What was the most important thing that happened in the 20th century?'
    results = searchFLOWhybrid(file_path, query, config)
    for chunk, score, sentiment in results:
        print(f"Chunk: {chunk['text']}")
        print("---")
        print(f"Start Char: {chunk['start_char']}")
        print(f"End Char: {chunk['end_char']}")
        source = chunk.get('source', 'No source available')
        print(f"Source: {source}")
        print(f"Relevance Score: {score:.1f}")
        print(f"Sentiment: {sentiment:.1f}")
        print("---")