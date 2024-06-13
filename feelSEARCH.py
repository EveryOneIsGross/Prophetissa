# ver 09
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from gensim.models import Word2Vec
import logging
import pickle
import os
from textblob import TextBlob
from sklearn.neighbors import KernelDensity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class feelsConfig:
    def __init__(self, topk_results=16, chunk_size=32, max_tokens=128):
        self.TOPK_RESULTS = topk_results
        self.CHUNK_SIZE = chunk_size
        self.MAXTOKENS = max_tokens

def preprocess_and_chunk(file_path, chunk_size=64, overlap=0.5, fallback_chunk_size=512):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()

    sentences = text.replace('\n', ' ').split('. ')

    chunks = []
    text_offset = 0

    if len(sentences) <= 1 and '.' not in text:
        logging.warning("No valid punctuation for sentence splitting detected, using fallback method.")
        words = text.split()
        num_chunks = max(1, len(words) // fallback_chunk_size)
        for i in range(0, len(words), fallback_chunk_size):
            chunk_text = ' '.join(words[i:i+fallback_chunk_size])
            start_char = text_offset
            end_char = start_char + len(chunk_text)
            chunks.append({
                'text': chunk_text,
                'sentences': [words[i:i+fallback_chunk_size]],
                'start_char': start_char,
                'end_char': end_char,
                'source': file_path
            })
            text_offset = end_char + 1
    else:
        words = [sentence.split() for sentence in sentences]
        step_size = max(1, int(chunk_size * (1 - overlap)))
        for i in range(0, len(words), step_size):
            chunk_words = [word for sentence in words[i:i + chunk_size] for word in sentence]
            chunk_text = ' '.join(chunk_words)
            start_char = text_offset
            end_char = start_char + len(chunk_text)
            chunks.append({
                'text': chunk_text,
                'sentences': words[i:i + chunk_size],
                'start_char': start_char,
                'end_char': end_char,
                'source': file_path
            })
            text_offset = end_char + 1

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

    top_chunks = [(chunks[i], similarities[i]) for i in top_indices]
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
    current_chunk = {'text': '', 'sentences': [], 'start_char': chunks[0]['start_char'], 'end_char': 0, 'source': chunks[0]['source']}
    for i in range(len(chunks)):
        current_chunk['text'] += chunks[i]['text'] + ' '
        current_chunk['sentences'].extend(chunks[i]['sentences'])
        current_chunk['end_char'] = chunks[i]['end_char']
        if len(current_chunk['sentences']) >= min_chunk_size and (
            len(current_chunk['sentences']) >= max_chunk_size or
            i == len(chunks) - 1 or
            abs(sentiments[i] - sentiments[i-1]) > sentiment_threshold or
            np.max(np.abs(density_map[i//len(density_map)] - density_map[(i-1)//len(density_map)])) > density_threshold
        ):
            adaptive_chunks.append(current_chunk)
            if i < len(chunks) - 1:
                current_chunk = {'text': '', 'sentences': [], 'start_char': chunks[i+1]['start_char'], 'end_char': 0, 'source': chunks[i+1]['source']}
    return adaptive_chunks

def load_or_process_data(file_path, config):
    cache_path = f"{file_path}_feel.pickle"
    if os.path.exists(cache_path):
        logging.info(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    else:
        logging.info("Processing data...")
        chunks = preprocess_and_chunk(file_path, chunk_size=config.CHUNK_SIZE)
        model = train_word2vec(chunks)
        sentiments = analyze_sentiment(chunks)
        corpus_vectors = [np.mean([model.wv[word] for word in chunk['text'].split() if word in model.wv], axis=0) for chunk in chunks if any(word in model.wv for word in chunk['text'].split())]
        smooth_corpus_vectors = smooth_vectors(np.array(corpus_vectors), sentiments)
        interpolation_points = np.linspace(0, 9, 50)
        interpolation_points = np.array(np.meshgrid(interpolation_points, interpolation_points)).T.reshape(-1, 2)
        density_map = semantic_density_mapping(smooth_corpus_vectors, interpolation_points, batch_size=1000)
        adaptive_chunks = adaptive_chunking(chunks, sentiments, density_map, min_chunk_size=5, max_chunk_size=config.MAXTOKENS)

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

def searchFLOWfeels(file_path, query, config):
    data = load_or_process_data(file_path, config)
    model = data['model']
    adaptive_chunks = data['adaptive_chunks']
    search_results = search_results_with_sentiment(model, adaptive_chunks, query, config, max_tokens=config.MAXTOKENS)
    return search_results

def search_results_with_sentiment(model, adaptive_chunks, query, config, max_tokens=50):
    results = semantic_search(model, query, adaptive_chunks, config.TOPK_RESULTS)
    unique_results = []
    seen_chunks = set()

    for chunk, score in results:
        if chunk['text'] not in seen_chunks:
            sentiment = TextBlob(chunk['text']).sentiment.polarity
            tokens = chunk['text'].split()
            if len(tokens) <= max_tokens:
                unique_results.append((chunk, round(score, 2), round(sentiment, 2)))
                seen_chunks.add(chunk['text'])
            else:
                shortened_chunk = {
                    'text': ' '.join(tokens[:max_tokens]),
                    'start_char': chunk['start_char'],
                    'end_char': chunk['start_char'] + len(' '.join(tokens[:max_tokens])),
                    'source': chunk['source']
                }
                if shortened_chunk['text'] not in seen_chunks:
                    sentiment = TextBlob(shortened_chunk['text']).sentiment.polarity
                    unique_results.append((shortened_chunk, round(score, 2), round(sentiment, 2)))
                    seen_chunks.add(shortened_chunk['text'])

    return unique_results

if __name__ == '__main__':
    config = feelsConfig(topk_results=2, chunk_size=32, max_tokens=512)
    file_path = 'conTEXTS\\text\\20th Century Women - The Script Lab1716430313.792667.txt'
    query = 'What was the most important thing that happened in the 20th century?'
    results = searchFLOWfeels(file_path, query, config)
    for chunk, score, sentiment in results:
        print(f"Chunk: {chunk['text']}")
        print(f"Start Char: {chunk['start_char']}")
        print(f"End Char: {chunk['end_char']}")
        print(f"Source: {chunk['source']}")
        print(f"Relevance Score: {score:.3f}")
        print(f"Sentiment: {sentiment:.3f}")
        print("---")


if __name__ == '__main__':
    main()
