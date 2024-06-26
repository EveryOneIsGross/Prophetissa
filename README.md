# Prophetissa
-------------
![transparent_image](https://github.com/EveryOneIsGross/Prophetissa/assets/23621140/9b742d09-b4e3-466a-b214-fb976a5031e3)

Prophetissa is an automated dataset generation tool designed to facilitate fine-tuning of AI models. It operates by taking input text and processing it through multiple search modules, each employing different techniques for text chunking, semantic search, and sentiment analysis. These search methods include hybrid, feel, PDF, wiki, and fractal search. The main script acts as a central coordinator, processing the results from these search modules, generating questions and answers based on the context, and saving the output in a structured JSON format. This pipeline ensures the generation of high-quality, contextually rich datasets suitable for training and fine-tuning AI models.

Each search module follows its own flow for processing the input text, performing search operations, and returning relevant results. The main script then processes these results, formats the context, generates questions and answers, and saves the data to a JSON file.

```
----------------------------------------------------------------------------
# ATTN:latest search additions have broken schemas for judgeDREAD (wip) 💀☕
Certain generations work ok, pays to seperate different search jsons for now
NAH ITS BROKEN, WORKING ON FIXING VALUES TO M DIST. FOR THRESHOLDING RESULTS
----------------------------------------------------------------------------
```

Prophetissa (prophetissa.py):
1. Loads the configuration and initializes the OpenAI client.
2. Defines utility functions for citation insertion, question generation, answer generation, and JSON data handling.
3. Processes search results based on the selected search tool (hybrid, feels, pdf2md, wikisearch, or fractal).
4. Formats the search results, generates questions and answers, and saves the data to a JSON file.

Search Modules:
1. feelSEARCH.py:
   - Preprocesses and chunks the input text.
   - Trains a Word2Vec model on the chunked text.
   - Performs sentiment analysis and semantic search using the trained model.
   - Adapts the chunk size based on sentiment and semantic density.

2. hybrid25SEARCH.py:
   - Splits the input text into sentences and embeds them using the Embed4All model.
   - Dynamically chunks the sentences based on the specified chunk size and threshold.
   - Performs semantic search with BM25 scoring, combining cosine similarity and BM25 scores.

3. pdfSEARCH.py:
   - Splits the input Markdown text into chunks based on pages, paragraphs, and sentences.
   - Extracts image filenames and descriptions from the Markdown.
   - Calculates keyword frequency and performs keyword search on the chunks.
   - Appends image descriptions to relevant chunks and calculates relevance scores.
  
   PDF md was generated using ['marker'](https://github.com/VikParuchuri/marker)
   Use ['luna.py'](https://github.com/EveryOneIsGross/Prophetissa/blob/main/luna.py) (ollama and moondream) to process generated .md to add image descriptions.

5. wikiSEARCH.py:
   - Fetches Wikipedia pages based on the input query.
   - Preprocesses and chunks the page content.
   - Expands the query using TextBlob, fuzzy matching, WordNet synonyms, and named entity recognition.
   - Analyzes the sentiment and keyword frequency of the page summary.

6. fractalSEARCH.py:
   - Preprocesses and chunks the input text.
   - Trains a Word2Vec model on the chunked text.
   - Performs Mandelbrot chunking based on the query vector and model.
   - Calculates similarities between the query vector and chunk vectors.
   - Returns the top-k most similar chunks with sentiment and relevance scores.
  
   You have to provide a 'folder' instead of 'file' when providing paths here, as fractal chunking ingests multiple documents to cluster.

Here's a Mermaid diagram illustrating the flow of the main script and its interaction with the search modules:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffaa00', 'primaryTextColor': '#000000', 'primaryBorderColor': '#ffaa00', 'lineColor': '#ffaa00', 'secondaryColor': '#ffaa00', 'tertiaryColor': '#ffaa00', 'clusterBkg': 'none', 'clusterBorder': 'none', 'fontSize': '20px'}}}%%
graph TD
    A[Main Script] --> B{Search Tool}
    B --> C{hybrid}
    B --> D{feels}
    B --> E{pdf2md}
    B --> F{wikisearch}
    B --> G{fractal}
    C --> H[hybrid25SEARCH]
    D --> I[feelSEARCH]
    E --> J[pdfSEARCH]
    F --> K[wikiSEARCH]
    G --> L[fractalSEARCH]
    H --> M[Process Search Results]
    I --> M
    J --> M
    K --> M
    L --> M
    M --> N[Format Context]
    N --> O[Generate Questions]
    O --> P[Generate Answers]
    P --> Q[Save to JSON]
```

Prompts are formatted ala mistrals cookbook.
~~
```
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge. Generate {num_questions_per_chunk}
questions based on the context. The questions should be diverse in nature across the
document. Restrict the questions to the context information provided.
```

```
Context information is below
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
answer the query.
Query: {generated_query_str}
Answer: 
```
[DOCS](https://docs.mistral.ai/guides/finetuning/)

TO RUN:
-------
```

~ensure yr seedcorpus.txt is formatted with items on new lines~

> python prophetissa.py --file_path "corpus.txt" --seed_queries_file "seedcorpus.txt" --output_file "dataset.json"

~ to rank results and create mistral compatible jsonl

> python judgeDREAD.py dataset.json --threshold 0.75

```

OUTPUT SCHEMA:
--------------

THIS IS A BIT OF A MESS, I GOT IMPATIENT SO THERE IS SOME MISALIGNED KEYS AND NESTING STILL THAT COULD BE REFACTORED TIGHTER.
The below schema is for pdf2md converted docs. The search generations have [] placeholders for fields not yet implemented.

```json
{
  "query": "string",
  "context": "string",
  "citations": [
    {
      "start_char": "string",
      "end_char": "string",
      "source": "string",
      "page_number": "string",
      "chunk_number": "integer",
      "line_number": "string"
    }
  ],
  "chunks": [
    {
      "text": "string",
      "start_char": "string",
      "end_char": "string",
      "page_number": "string",
      "line_number": "string",
      "image_filenames": [
        "string"
      ],
      "image_descriptions": [
        "string"
      ],
      "sentiment": "float",
      "relevance_score": "float"
    }
  ],
  "question": "string",
  "answer": "string"
}

```
FLOW:
-----
```
+--------------------------------------------------------------------------------+
| Code Overview:                                                                 |
+--------------------------------------------------------------------------------+
|                                                                                |
| 1. Import Libraries                                                            |
|    - JSON for handling data serialization.                                     |
|    - Custom modules for search functionality and the Ollama inference API.     |
|                                                                                |
| 2. Ollama Client Initialization                                                |
|    - hijack openai api with yr local llm                                       |
|                                                                                |
| 3. Function Definitions:                                                       |
|    - generate_questions                                                        |
|      * Generates synthetic questions based on a provided context using the     |
|        Ollama inference API. Questions are designed to be diverse and relevant |
|        to the input text, enhancing dataset richness for training.             |
|                                                                                |
|    - generate_answer                                                           |
|      * For each synthetic question, generates a grounded answer using the      |
|        same context. These Q&A pairs serve as a foundational training dataset  |
|        for models requiring contextual understanding.                          |
|                                                                                |
|    - load_json_data                                                            |
|      * Loads data from a JSON file, used for appending new Q&A pairs.          |
|                                                                                |
|    - save_json_data                                                            |
|      * Serializes and saves updated Q&A pairs to a JSON file, preserving       |
|        the data for further model training.                                    |
|                                                                                |
|    - process_search_results                                                    |
|      * Utilizes a semantic search custom module to fetch relevant text chunks  |
|        from a provided corpus.                                                 |
|      * Converts these chunks into a structured context which is then used to   |
|        generate synthetic Q&A pairs, simulating realistic user interactions    |
|        and enhancing the training dataset's diversity and quality.             |
|                                                                                |
| 4. Main Execution Flow:                                                        |
|    - Reads a list of queries from a file.                                      |
|    - Each query is processed to simulate a real-world scenario where the model |
|      would need to understand and interact based on textual inputs.            |
|    - The results, including synthetic Q&A pairs, are saved to a JSON file,     |
|      ready for use in training scenarios.                                      |
|                                                                                |
+--------------------------------------------------------------------------------+
|                                                                                |
| Note:                                                                          |
| The main functionality focuses on generating synthetic data through            |
| interaction with the Ollama inference model, tailored to create a robust       |
| training dataset that mimics real-world informational needs and responses.     |
|                                                                                |
+--------------------------------------------------------------------------------+
```



DEPENDENCIES:
-------------
```
python - whatever ver. you p don't have installed
mostly useless imports i neglected to scrub

warnings
logging
numpy
sklearn
pickle
gensim
os
textblob
openai
json
```
uses fine tuning prompt template from : [mistral ft guide](https://github.com/mistralai/mistral-finetune)
uses densefeelsSEARCH for retrieval.
uses ollama for inference. 


HOW densefeelSEARCH WORKS:
-------------------------

The script provided outlines a comprehensive text analysis pipeline that integrates various natural language processing (NLP) techniques. It starts with preprocessing text, trains a Word2Vec model for semantic representation, uses sentiment analysis, performs semantic density mapping, and adapts text chunking accordingly, eventually utilizing these processed chunks for semantic search. Here’s how each component works together and influences the final search results:

1. The script reads a text file, preprocesses it by dividing the text into manageable chunks based on sentence boundaries or fallbacks to fixed word counts if no sentences are detected. This initial chunking is critical as it sets the base units of text that will be further processed and analyzed.

2. These chunks are then used to train a Word2Vec model. This model learns semantic representations of words based on their contextual usage in the chunks, producing vectors that capture the meanings of words and phrases.

3. Each chunk is analyzed for sentiment using TextBlob, which assigns a polarity score reflecting the emotional tone of the text in each chunk. This sentiment data is used later in the adaptive chunking process.

4. Parallel to sentiment analysis, the corpus vectors (derived from Word2Vec) of each chunk are used to generate a semantic density map via kernel density estimation. This map indicates areas of high and low semantic concentration across the corpus, which can guide the adaptive chunking process.

5. Chunks are then dynamically adjusted based on several factors:
- **Size constraints** (minimum and maximum sizes for chunks)
- **Sentiment shifts** (changes in sentiment that exceed a predefined threshold)
- **Semantic density changes** (fluctuations in semantic density that surpass a set threshold, derived from the density map)
This process ensures that each chunk is informationally coherent and contextually complete, which is crucial for maintaining the integrity of semantic analysis.

6. Once the adaptive chunks are finalized, a semantic search is conducted for a given query. The search leverages the Word2Vec model to find chunks that semantically align with the query by comparing vector similarities. Chunks are ranked based on their cosine similarity to the query vector, and the top results are selected based on configuration settings (top-k results).

7. n Results returned. 

Because chunks are adapted based on sentiment and semantic density, they are likely to be more topically coherent and contextually relevant. This quality directly enhances the efficacy of semantic searches since the search algorithm can operate on chunks that accurately represent distinct ideas or themes. Adaptive chunking, influenced by semantic density and sentiment, ensures that the text chunks used in the search are optimally configured to respond to the query. This leads to results that are not only relevant but also rich in contextual integrity, making them more useful for users. The inclusion of sentiment scores in the search results adds an additional layer of information, allowing users to gauge not just the relevance but also the emotional tone of the content related to their queries.

In conclusion, the integration of semantic density mapping into the adaptive chunking process significantly enhances the overall text analysis pipeline by ensuring that the chunks are both semantically and emotionally coherent. This coherence directly influences the quality of the semantic search results, making the system robust for applications requiring nuanced text understanding and retrieval.

The below graphs only sort of make sense, but opus is in cooldown for me and chatgpt can't graph for ****
```
                               +------------+
                               | Text File  |
                               +-----+------+
                                     |
                                     |
                               +-----v------+
                               | Preprocess |
                               |  & Chunk   |
                               +------------+----------+
                                                       |
                                                       |
+------------------+          +------------+           +-------------------+
| Query            |          | Semantic   |           | Train Word2Vec    |
|                  +--------->+ Search     +--------+  |                   |
| (A2)             |          | (K1)       |        |  +-------------------+
+------------------+          +------------+        |              |
                                                    v              v
                                       +-------------+------+     +--+---------------+
                                       | Analyze            |     | Corpus           |
                                       | Sentiment (D1)     |     | Vectors (E1)     |
                                       +----------+---------+     +---------+--------+
                                                  |                           |
                                                  v                           v
                                      +-----------+-----------+-+-------------+-----+
                                      | Smooth Vectors (F1)   | | Smooth            |
                                      |                       | | Corpus Vectors    |
                                      +-----------+-----------+ +---------+---------+
                                                  |                           |
                                                  v                           v
                                    +-------------+-------------+ +------------+--------------+
                                    | Interpolation              | | Semantic                  |
                                    | Points (H1)                | | Density Mapping (I1)      |
                                    +-------------+--------------+ +------------+--------------+
                                                  |                             |
                                                  |                             |
                                                  +-------------+---------------+
                                                                |
                                                                v
                                               +----------------+---------------+
                                               | Density Map (J1)               |
                                               +----------------+---------------+
                                                                |
                                                                v
                                               +----------------+--------------+
                                               | Adaptive Chunking (L1)        |
                                               +-------------------------------+
                                                                |
                                                                v
                                               +----------------+--------------+
                                               | Final Output:                 |
                                               | Search Results with Sentiment |
                                               +-------------------------------+

```
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffaa00', 'primaryTextColor': '#000000', 'primaryBorderColor': '#ffaa00', 'lineColor': '#ffaa00', 'secondaryColor': '#ffaa00', 'tertiaryColor': '#ffaa00', 'clusterBkg': 'none', 'clusterBorder': 'none', 'fontSize': '20px'}}}%%
graph TD
A((Input: Queries and Corpus)) --> B1[Preprocess and Chunk]
B1 --> C1[Train Word2Vec]
B1 --> D1[Analyze Sentiment]
C1 --> E1[Corpus Vectors]
D1 --> E1
E1 --> F1[Smooth Vectors]
F1 --> G1[Smooth Corpus Vectors]
G1 --> H1[Interpolation Points]
G1 --> I1[Semantic Density Mapping]
H1 --> I1
I1 --> J1[Density Map]
J1 --> K1[Adaptive Chunking]
K1 --> L1[Adaptive Chunks]
A --> M1[Process Search Results]
M1 --> L1
L1 --> N1[Generate Questions]
N1 --> O1[Generate Answers]
O1 --> P1[Load JSON Data]
O1 --> Q1[Save JSON Data]
P1 <--> R1[Main Execution Flow]
Q1 <--> R1
R1 --> S((Final Output: Training Dataset))

subgraph Preprocessing
B1[Preprocess and Chunk]
end

subgraph Word2Vec
C1[Train Word2Vec]
E1[Corpus Vectors]
end

subgraph SentimentAnalysis
D1[Analyze Sentiment]
end

subgraph Smoothing
F1[Smooth Vectors]
G1[Smooth Corpus Vectors]
end

subgraph DensityMapping
H1[Interpolation Points]
I1[Semantic Density Mapping]
J1[Density Map]
end

subgraph AdaptiveChunking
K1[Adaptive Chunking]
L1[Adaptive Chunks]
end

subgraph Search
M1[Process Search Results]
end

subgraph GenerateData
N1[Generate Questions]
O1[Generate Answers]
end

subgraph JSONHandling
P1[Load JSON Data]
Q1[Save JSON Data]
end

subgraph ExecutionFlow
R1[Main Execution Flow]
S((Final Output: Training Dataset))
end

```
