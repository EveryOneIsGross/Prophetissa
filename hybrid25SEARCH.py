# v 01

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import logging
import pickle
import os
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
from gpt4all import Embed4All

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, topk_results=16, max_tokens=128):
        self.TOPK_RESULTS = topk_results
        self.MAXTOKENS = max_tokens

def preprocess_and_chunk(file_path, chunk_size=128, overlap=0.5):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()

    chunks = []
    sentences = text.split('.')

    for sentence in sentences:
        if '\n' in sentence:
            sub_sentences = sentence.split('\n')
            for sub_sentence in sub_sentences:
                sub_sentence = sub_sentence.strip()
                if sub_sentence:
                    chunks.append({'text': sub_sentence, 'words': sub_sentence.split()})
        else:
            sentence = sentence.strip()
            if sentence:
                words = sentence.split()
                num_words = len(words)

                for i in range(0, num_words, int(chunk_size * (1 - overlap))):
                    start_index = i
                    end_index = min(i + chunk_size, num_words)
                    chunk_words = words[start_index:end_index]
                    chunk_text = ' '.join(chunk_words)
                    chunks.append({'text': chunk_text, 'words': chunk_words})

    return chunks

def compute_bm25(tf, df, N, dl, avdl, k1=1.5, b=0.75):
    if (N - df + 0.5) / (df + 0.5) + 1 <= 0:
        return 0
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    return idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * (dl / avdl))))

def semantic_search_with_bm25(embeddings, query_embedding, chunks, top_k, query):
    documents = [chunk['text'] for chunk in chunks]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    total_docs = len(documents)
    doc_lengths = X.sum(axis=1)
    avg_dl = doc_lengths.mean()

    query_vec = vectorizer.transform([query]).toarray()
    bm25_scores = np.zeros(X.shape[0])

    for i, j in zip(*X.nonzero()):
        bm25_scores[i] += compute_bm25(X[i, j], np.log((total_docs - np.sum(X[:, j] > 0) + 0.5) / (np.sum(X[:, j] > 0) + 0.5)), total_docs, doc_lengths[i, 0], avg_dl)
    # convert bm25 scores to a probability distribution
    bm25_scores = np.exp(bm25_scores) / np.sum(np.exp(bm25_scores))
    cosine_scores = cosine_similarity([query_embedding], embeddings).flatten()
    final_scores = bm25_scores * cosine_scores

    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    return [(chunks[i]['text'], final_scores[i]) for i in top_indices]

def load_or_process_data(file_path, embedder):
    cache_path = f"{file_path}.pickle"
    if os.path.exists(cache_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    else:
        logging.info("Processing data...")
        chunks = preprocess_and_chunk(file_path)
        embeddings = [embedder.embed(chunk['text']) for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        data = {
            'embeddings': embeddings,
            'chunks': chunks,
            'documents': documents
        }
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logging.info(f"Data processed and cached: {cache_path}")
        return data

def main(file_path, query, config):
    embedder = Embed4All('nomic-embed-text-v1.f16.gguf')
    data = load_or_process_data(file_path, embedder)
    query_embedding = embedder.embed(query)
    search_results = semantic_search_with_bm25(data['embeddings'], query_embedding, data['chunks'], config.TOPK_RESULTS, query)
    
    results_with_sentiment = []
    for chunk, score in search_results:
        sentiment = TextBlob(chunk).sentiment.polarity
        results_with_sentiment.append((chunk, score, sentiment))
    
    return results_with_sentiment

if __name__ == '__main__':
    config = Config()
    file_path = 'INPUT_PROMPTS\cyberanimism.txt'
    query = 'What is animism?'
    results = main(file_path, query, config)
    for text, score, sentiment in results:
        print(f"Chunk: {text}")
        print(f"Relevance Score: {score:.3f}")
        print(f"Sentiment: {sentiment:.3f}")
        print("---")
