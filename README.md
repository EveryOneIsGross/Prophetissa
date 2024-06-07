# Prophetissa
RAG dataset generator using ollama and emo vector search.
Currently only using .txt for the injests as proof of concept.

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
---

uses fine tuning prompt template from : [mistral ft guide](https://github.com/mistralai/mistral-finetune)
uses densefeelsSEARCH for retrieval.

```
                               +------------+
                               | Text File  |
                               +-----+------+
                                     |
                                     |
                               +-----v------+
                               | Preprocess |
                               |  & Chunk   |
                               +-----+------+
                                     |
                                     v
+------------------+          +-----+------+           +-------------------+
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
                                      +-----------+-----------+ +-------------+-----+
                                      | Smooth Vectors (F1)   | | Smooth            |
                                      |                       | | Corpus Vectors    |
                                      +-----------+-----------+ +---------+--------+
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
                                               | Density Map (J1)              |
                                               +----------------+---------------+
                                                                |
                                                                v
                                               +----------------+---------------+
                                               | Adaptive Chunking (L1)        |
                                               +-------------------------------+
                                                                |
                                                                v
                                               +----------------+---------------+
                                               | Final Output:                 |
                                               | Search Results with Sentiment |
                                               +-------------------------------+

```
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffaa00', 'primaryTextColor': '#ffaa00', 'primaryBorderColor': '#ffaa00', 'lineColor': '#ffaa00', 'secondaryColor': '#ffaa00', 'tertiaryColor': '#ffaa00', 'clusterBkg': 'none', 'clusterBorder': 'none', 'fontSize': '0px'}}}%%
graph TD
A1((Text File)) --> B1[Preprocess and Chunk]
A2((Query)) --> K1[Semantic Search]

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
J1 --> L1[Adaptive Chunking]

L1 --> K1
K1 --> M1[Search Results with Sentiment]

M1 --> N1((Final Output))

subgraph Input
A1((Text File))
A2((Query))
end

subgraph Preprocess
B1[Preprocess and Chunk]
end

subgraph Word2Vec
C1[Train Word2Vec]
E1[Corpus Vectors]
end

subgraph Sentiment
D1[Analyze Sentiment]
end

subgraph Smooth
F1[Smooth Vectors]
G1[Smooth Corpus Vectors]
end

subgraph Density
H1[Interpolation Points]
I1[Semantic Density Mapping]
J1[Density Map]
end

subgraph Adaptive
L1[Adaptive Chunking]
end

subgraph Search
K1[Semantic Search]
M1[Search Results with Sentiment]
end

subgraph Output
N1((Final Output))
end


```
