# ver 07
import warnings
import numpy as np
from gensim.models import Word2Vec
import logging
import pickle
import os
from textblob import TextBlob
from sklearn.neighbors import KernelDensity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Config:
    def __init__(self, topk_results=16, max_tokens=128):
        self.TOPK_RESULTS = topk_results
        self.MAXTOKENS = max_tokens

def preprocess_and_chunk(file_path, chunk_size=64, overlap=0.5, fallback_chunk_size=64):
    # Read a text file and preprocess it into fixed-size chunks with overlap.
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()

    # Try to split by punctuation
    sentences = text.replace('\n', ' ').split('. ')

    if len(sentences) <= 1 and '.' not in text:  # Check if there are no valid sentence delimiters
        logging.warning("No valid punctuation for sentence splitting detected, using fallback method.")
        # Fallback: Split by fixed number of words if no sentences are detected
        words = text.split()
        chunks = []
        num_chunks = max(1, len(words) // fallback_chunk_size)
        for i in range(0, len(words), fallback_chunk_size):
            chunk_text = ' '.join(words[i:i+fallback_chunk_size])
            chunks.append({'text': chunk_text, 'sentences': [words[i:i+fallback_chunk_size]]})
    else:
        # Regular sentence-based chunking
        words = [sentence.split() for sentence in sentences]
        chunks = []
        step_size = max(1, int(chunk_size * (1 - overlap)))  # Ensure step size is at least 1
        for i in range(0, len(words), step_size):
            chunk_words = [word for sentence in words[i:i + chunk_size] for word in sentence]
            chunk_text = ' '.join(chunk_words)
            chunks.append({'text': chunk_text, 'sentences': words[i:i + chunk_size]})

    return chunks



def train_word2vec(chunks, vector_size=100, window=5, min_count=1, workers=4):
    sentences = [sentence for chunk in chunks for sentence in chunk['sentences']]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

def analyze_sentiment(chunks):
    sentiments = [TextBlob(chunk['text']).sentiment.polarity for chunk in chunks]
    return np.array(sentiments)

def smooth_vectors(corpus_vectors, sentiments, window_size=5):
    weighted_vectors = corpus_vectors * sentiments.reshape(-1, 1)
    smoothed_vectors = np.zeros_like(corpus_vectors)
    for i in range(len(corpus_vectors)):
        start = max(0, i - window_size // 2)
        end = min(len(corpus_vectors), i + window_size // 2 + 1)
        smoothed_vectors[i] = np.mean(weighted_vectors[start:end], axis=0)
    return smoothed_vectors

def semantic_search(model, query, chunks, top_k):
    query_words = query.split()
    query_vectors = [model.wv[word] for word in query_words if word in model.wv]

    if not query_vectors:
        return []

    query_vector = np.mean(query_vectors, axis=0)
    chunk_vectors = [np.mean([model.wv[word] for word in chunk['text'].split() if word in model.wv], axis=0) for chunk in chunks]
    similarities = [np.dot(query_vector, chunk_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vec)) for chunk_vec in chunk_vectors]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    top_chunks = [(chunks[i]['text'], similarities[i]) for i in top_indices]
    return top_chunks

def semantic_density_mapping(corpus_vectors, interpolation_points, batch_size=1000):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(corpus_vectors)
    grid_points = np.array(np.meshgrid(interpolation_points[:, 0], interpolation_points[:, 1])).T.reshape(-1, 2)

    density_map = np.zeros((len(interpolation_points), len(interpolation_points)))
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_vectors = np.hstack([batch_points, np.zeros((len(batch_points), corpus_vectors.shape[1] - 2))])
        batch_density = np.exp(kde.score_samples(batch_vectors))

        start_row = i // len(interpolation_points)
        end_row = min((i + batch_size) // len(interpolation_points), len(interpolation_points))
        start_col = i % len(interpolation_points)
        end_col = min(start_col + batch_size, len(interpolation_points))

        rows = end_row - start_row
        cols = end_col - start_col
        density_map[start_row:end_row, start_col:end_col] = batch_density[:rows * cols].reshape(rows, cols)

    return density_map

def adaptive_chunking(chunks, sentiments, density_map, min_chunk_size, max_chunk_size, sentiment_threshold=0.2, density_threshold=0.1):
    adaptive_chunks = []
    current_chunk = {'text': '', 'sentences': []}
    for i in range(len(chunks)):
        current_chunk['text'] += chunks[i]['text'] + ' '
        current_chunk['sentences'].extend(chunks[i]['sentences'])
        if len(current_chunk['sentences']) >= min_chunk_size and (
            len(current_chunk['sentences']) >= max_chunk_size or
            i == len(chunks) - 1 or
            abs(sentiments[i] - sentiments[i-1]) > sentiment_threshold or
            np.max(np.abs(density_map[i//len(density_map)] - density_map[(i-1)//len(density_map)])) > density_threshold
        ):
            adaptive_chunks.append(current_chunk)
            current_chunk = {'text': '', 'sentences': []}
    return adaptive_chunks

def load_or_process_data(file_path):
    cache_path = f"{file_path}.pickle"
    if os.path.exists(cache_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    else:
        logging.info("Processing data...")
        chunks = preprocess_and_chunk(file_path, chunk_size=16)
        model = train_word2vec(chunks)
        sentiments = analyze_sentiment(chunks)
        corpus_vectors = [np.mean([model.wv[word] for word in chunk['text'].split() if word in model.wv], axis=0) for chunk in chunks if any(word in model.wv for word in chunk['text'].split())]
        smooth_corpus_vectors = smooth_vectors(np.array(corpus_vectors), sentiments)
        interpolation_points = np.linspace(0, 9, 50)
        interpolation_points = np.array(np.meshgrid(interpolation_points, interpolation_points)).T.reshape(-1, 2)
        density_map = semantic_density_mapping(smooth_corpus_vectors, interpolation_points, batch_size=1000)
        adaptive_chunks = adaptive_chunking(chunks, sentiments, density_map, min_chunk_size=5, max_chunk_size=20)

        data = {
            'model': model,
            'chunks': chunks,
            'sentiments': sentiments,
            'smooth_corpus_vectors': smooth_corpus_vectors,
            'adaptive_chunks': adaptive_chunks,
            'density_map': density_map
        }

        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logging.info(f"Data processed and cached: {cache_path}")
        return data

def main(file_path, query, config):
    data = load_or_process_data(file_path)
    model = data['model']
    adaptive_chunks = data['adaptive_chunks']
    search_results = search_results_with_sentiment(model, adaptive_chunks, query, config)
    return search_results

def search_results_with_sentiment(model, adaptive_chunks, query, config):
    results = semantic_search(model, query, adaptive_chunks, config.TOPK_RESULTS)
    unique_results = []
    seen_chunks = set()

    for chunk, score in results:
        if chunk not in seen_chunks:
            sentiment = TextBlob(chunk).sentiment.polarity
            tokens = chunk.split()
            if len(tokens) <= config.MAXTOKENS:
                unique_results.append((chunk, score, sentiment))
                seen_chunks.add(chunk)
            else:
                shortened_chunk = ' '.join(tokens[:config.MAXTOKENS])
                if shortened_chunk not in seen_chunks:
                    sentiment = TextBlob(shortened_chunk).sentiment.polarity
                    unique_results.append((shortened_chunk, score, sentiment))
                    seen_chunks.add(shortened_chunk)

    return unique_results

if __name__ == '__main__':
    main()
