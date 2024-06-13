import wikipedia
import markdownify
from fuzzywuzzy import process
from textblob import TextBlob
import numpy as np
import logging

# Set Wikipedia to use the English language
wikipedia.set_lang('en')

class WikiSearchConfig:
    def __init__(self, n_chunks=1, chunk_size=32, max_tokens=128, topk_results=1):
        self.N_CHUNKS = n_chunks
        self.CHUNK_SIZE = chunk_size
        self.MAXTOKENS = max_tokens
        self.TOPK_RESULTS = topk_results

def preprocess_and_chunk(page_content, config):
    # Split the content into chunks
    words = page_content.split()
    chunks = [' '.join(words[i:i + config.CHUNK_SIZE]) for i in range(0, len(words), config.CHUNK_SIZE)]
    return chunks

def fetch_wikipedia_page(query):
    try:
        page = wikipedia.page(query)
        # Specify the HTML parser explicitly
        page_content = markdownify.markdownify(page.content, parser='lxml.parser')
        return {
            'title': page.title,
            'summary': page.summary,
            'content': page_content,
            'url': page.url
        }
    except wikipedia.DisambiguationError as e:
        print(f"Disambiguation error for query '{query}': {e.options}")
        return None
    except wikipedia.PageError:
        print(f"Page error for query '{query}'.")
        return None

def expand_query(query):
    # Extract terms using TextBlob
    blob = TextBlob(query)
    extracted_terms = list(set(blob.noun_phrases))
    
    # Include original query and handle acronyms
    all_terms = set([query])
    if len(query) < 4:
        all_terms.add(query.upper())
        period_separated = '.'.join(query)
        all_terms.add(period_separated)
    
    for term in extracted_terms:
        all_terms.add(term)
        if len(term) < 4:
            all_terms.add(term.upper())
    
    # Perform fuzzy matching and combine the results
    fuzzy_matches = []
    for term in all_terms:
        matches = process.extract(term, all_terms, limit=5)
        fuzzy_matches.extend([match[0] for match in matches])
    
    return ' '.join(set(fuzzy_matches))

def analyze_summary(summary, query):
    blob = TextBlob(summary)
    polarity = np.round(blob.sentiment.polarity, 3)
    subjectivity = np.round(blob.sentiment.subjectivity, 3)
    keyword_freq = np.float64(sum(summary.lower().count(term.lower()) for term in query.split())) / len(summary.split())
    return polarity, subjectivity, keyword_freq

def searchFLOWwiki(query, config):
    search_results = wikipedia.search(query, results=config.TOPK_RESULTS)
    if not search_results:
        return None
    
    results = []
    for result in search_results:
        try:
            data = fetch_wikipedia_page(result)
            if data is None:
                continue
            chunks = preprocess_and_chunk(data['content'], config)
            result_chunks = []
            for i, chunk in enumerate(chunks[:config.N_CHUNKS]):
                start_char = i * config.CHUNK_SIZE
                end_char = min((i + 1) * config.CHUNK_SIZE, len(data['content']))
                result_chunks.append({
                    'text': chunk,
                    'start_char': start_char,
                    'end_char': end_char
                })
            polarity, subjectivity, keyword_freq = analyze_summary(data['summary'], query)
            results.append({
                'title': data['title'],
                'summary': data['summary'],
                'search_results': result_chunks,
                'url': data['url'],
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'keyword_frequency': np.round(keyword_freq, 3)
            })
        except wikipedia.PageError:
            continue
    
    return results

if __name__ == '__main__':
    # Test the search module independently
    config = WikiSearchConfig(n_chunks=1, chunk_size=256, max_tokens=512, topk_results=3)
    initial_query = 'ai'
    
    # Expand the initial query
    combined_query = expand_query(initial_query)
    print("Combined Query:", combined_query)
    
    # Search using the combined query
    print(f"\nSearching for: {combined_query}")
    results = searchFLOWwiki(combined_query, config)
    if results:
        for idx, result in enumerate(results):
            print(f"\nResult {idx + 1}:")
            print("Title:", result['title'])
            print("Summary:", result['summary'])
            print("URL:", result['url'])
            print("Sentiment Polarity:", result['sentiment_polarity'])
            print("Sentiment Subjectivity:", result['sentiment_subjectivity'])
            print("Keyword Frequency:", result['keyword_frequency'])
            print("Search Results:")
            for i, chunk in enumerate(result['search_results']):
                print(f"Chunk {i + 1}:")
                print("Text:", chunk['text'])
                print("Start Char:", chunk['start_char'])
                print("End Char:", chunk['end_char'])
                print()
    else:
        print(f"No results for query: {combined_query}")
