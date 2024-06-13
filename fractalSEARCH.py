# v 03
import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
import pickle
import logging
from textblob import TextBlob
import hashlib

class fractalSearchConfig:
    def __init__(self, topk_results=8, initial_chunk_size=2, max_iter=8, min_chunk_size=1):
        self.TOPK_RESULTS = topk_results
        self.INITIAL_CHUNK_SIZE = initial_chunk_size
        self.MAX_ITER = max_iter
        self.MIN_CHUNK_SIZE = min_chunk_size

class SemanticGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, data=None):
        self.graph.add_node(node_id, data=data)

    def add_edge(self, node1_id, node2_id, weight=1.0):
        self.graph.add_edge(node1_id, node2_id, weight=weight)

    def get_graph(self):
        return self.graph

def read_document(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def preprocess_documents(document_files):
    documents = []
    for file_path in document_files:
        document = read_document(file_path)
        if document:
            documents.append((file_path, document))
    return documents

def mandelbrot_chunking(sentences, chunk_size, query_vector, model, max_iter, min_chunk_size, start_offset=0):
    if len(sentences) <= min_chunk_size or max_iter == 0:
        chunk_text = ' '.join(sentences)
        return [(chunk_text, start_offset, start_offset + len(chunk_text))]

    chunks = []
    current_chunk = []
    current_offset = start_offset

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_vector = sentence_vector(chunk_text, model)
            divergence = calculate_divergence(query_vector, chunk_vector)
            if divergence > 2:
                sub_chunks = mandelbrot_chunking(current_chunk, chunk_size // 2, query_vector, model, max_iter - 1, min_chunk_size, current_offset)
                chunks.extend(sub_chunks)
            else:
                chunks.append((chunk_text, current_offset, current_offset + len(chunk_text)))
            current_offset += len(chunk_text) + 1  # +1 for the space between sentences
            current_chunk = []

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append((chunk_text, current_offset, current_offset + len(chunk_text)))

    return chunks

def calculate_divergence(query_vector, chunk_vector):
    return np.linalg.norm(query_vector - chunk_vector)

def train_and_save_model(documents, model_path):
    tokenized_documents = [doc.split() for _, doc in documents]
    model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)
    model.wv.save(model_path)

def load_model(model_path):
    return KeyedVectors.load(model_path, mmap='r')

def sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def generate_cache_key(document_folder):
    file_hashes = []
    for file_name in os.listdir(document_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(document_folder, file_name)
            with open(file_path, 'rb') as file:
                file_content = file.read()
                file_hash = hashlib.md5(file_content).hexdigest()
                file_hashes.append(file_hash)
    cache_key = '_'.join(file_hashes)
    return cache_key

def load_or_process_data(document_folder, config):
    cache_key = generate_cache_key(document_folder)
    cache_path = f"processed_data_{cache_key}.pickle"
    model_path = f"word2vec_model_{cache_key}.pkl"

    if os.path.exists(cache_path) and os.path.exists(model_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            data = pickle.load(cache_file)
        model = load_model(model_path)
        return data, model
    else:
        logging.info("Processing data...")
        document_files = [os.path.join(document_folder, file) for file in os.listdir(document_folder) if file.endswith('.txt')]
        documents = preprocess_documents(document_files)
        if not os.path.exists(model_path):
            train_and_save_model(documents, model_path)
        model = load_model(model_path)
        data = {
            'documents': documents,
            'model_path': model_path
        }
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logging.info(f"Data processed and cached: {cache_path}")
        return data, model

def searchFLOWfractals(query, document_folder, config):
    data, model = load_or_process_data(document_folder, config)
    documents = data['documents']
    semantic_graph = SemanticGraph()
    query_vector = sentence_vector(query, model)

    chunked_documents = []
    for file_path, doc in documents:
        sentences = sent_tokenize(doc)
        chunks = mandelbrot_chunking(sentences, config.INITIAL_CHUNK_SIZE, query_vector, model, config.MAX_ITER, config.MIN_CHUNK_SIZE)
        chunked_documents.extend([(file_path, chunk_text, start_char, end_char) for chunk_text, start_char, end_char in chunks])

    chunk_ids = [chunk_text for _, chunk_text, _, _ in chunked_documents]

    for file_path, chunk_text, start_char, end_char in chunked_documents:
        semantic_graph.add_node(chunk_text, data={'file_path': file_path, 'start_char': start_char, 'end_char': end_char})

    semantic_graph.add_node("query")

    chunk_vectors = [sentence_vector(chunk_text, model) for chunk_text in chunk_ids]
    similarities = []

    for chunk_vector in chunk_vectors:
        query_norm = np.linalg.norm(query_vector)
        chunk_norm = np.linalg.norm(chunk_vector)
        if query_norm == 0 or chunk_norm == 0:
            similarity = 0
        else:
            similarity = np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm)
        similarities.append(similarity)

    for i, chunk1_id in enumerate(chunk_ids):
        semantic_graph.add_edge("query", chunk1_id, weight=similarities[i])

    top_k_indices = np.argsort(similarities)[-config.TOPK_RESULTS:][::-1]
    top_k_chunks = [chunked_documents[i] for i in top_k_indices]

    results_with_sentiment_and_relevance = []
    for file_path, chunk_text, start_char, end_char in top_k_chunks:
        sentiment = round(TextBlob(chunk_text).sentiment.polarity, 3)
        relevance_score = similarities[chunk_ids.index(chunk_text)]
        results_with_sentiment_and_relevance.append((file_path, chunk_text, start_char, end_char, sentiment, relevance_score))

    return results_with_sentiment_and_relevance, semantic_graph.get_graph()

if __name__ == '__main__':
    query = "Where in the future?"
    document_folder = "inputs"
    config = fractalSearchConfig()

    top_k_results, semantic_graph = searchFLOWfractals(query, document_folder, config)

    print(f"Query: {query}")
    print("Top-k most similar results:")
    for result in top_k_results:
        print(f"Chunk: {result[1]}\nSource: {result[0]}\nStart Char: {result[2]}\nEnd Char: {result[3]}\nSentiment: {result[4]}\nRelevance Score: {result[5]}\n")
