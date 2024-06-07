# v 07
import json
import argparse
from openai import OpenAI
import re
from densefeelSEARCH import main, Config
#from hybrid25SEARCH import main, Config

TOPK = 8
MAXTOKENS = 128
QUESTIONMODEL = "qwen2:0.5b-instruct-fp16"
ANSWERMODEL = "qwen2:0.5b-instruct-fp16"


# Initialize the OpenAI client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

def generate_questions(context, num_questions):
    prompt = f"""
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, generate {num_questions}
questions based on the context. The questions should be diverse in nature across the
document. Restrict the questions to the context information provided. Each question should be a single sentence ending with a question mark or a newline character.
"""
    response = client.chat.completions.create(
        model=QUESTIONMODEL,
        messages=[
            {"role": "system", "content": "You are a curious assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_text = response.choices[0].message.content.strip()
    generated_questions = [q.strip() for q in generated_text.replace('?', '?\n').split('\n') if q.strip()]
    generated_questions = [q + '?' if not q.endswith('?') else q for q in generated_questions]
    # remove number bullet points numerals followed by a dot
    generated_questions = [re.sub(r'^\d+\.', '', q) for q in generated_questions]
    # remove leading characters and spaces before the first Capital letter in the question if available otheriwse next letter
    generated_questions = [q[q.find(q[0]):] for q in generated_questions] 
     
    print(prompt)
    print(generated_questions[0:3])
    # return only 3 questions
    return generated_questions[0:3]

def generate_answer(context, question):
    prompt = f"""
Context information is below
---------------------
{context}
---------------------
Given the context information and not prior knowledge,
answer the query.
Query: {question}
Answer:
"""
    response = client.chat.completions.create(
        model=ANSWERMODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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

def save_json_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def process_search_results(file_path, query, output_file):
    # Define the configuration for your search engine
    searchCONFIG = Config(topk_results=TOPK, max_tokens=MAXTOKENS)

    # Perform semantic search using your search engine
    search_results = main(file_path, query, searchCONFIG)

    # Skip generation if the search results are empty
    if not search_results:
        print(f"No search results found for query: {query}")
        return

    # Load existing JSON data from the output file
    existing_data = load_json_data(output_file)

    # Accumulate all search results into a single context
    context = ""
    chunks = []
    combined_score = 0
    for chunk, relevance_score, sentiment in search_results:
        formatted_chunk = f"Chunk: '{chunk}'\nRelevance Score: {relevance_score}\nSentiment: {sentiment}\n\n"
        context += formatted_chunk
        chunks.append({
            "text": chunk,
            "relevance_score": float(relevance_score),  # Convert to regular float
            "sentiment": sentiment
        })
        combined_score += float(relevance_score)  # Convert to regular float

    # Generate questions based on the accumulated context
    num_questions = 3  # Specify the number of questions to generate
    questions = generate_questions(context, num_questions)

    # Generate answers for each question and save the results
    for question in questions:
        answer = generate_answer(context, question)
        result = {
            "query": query,
            "context": context,
            "chunks": chunks,
            "combined_score": float(combined_score),  # Convert to regular float
            "question": question,
            "answer": answer            
        }
        existing_data.append(result)
        save_json_data(output_file, existing_data)
        print(f"Search results saved to {output_file}")

    # Save the updated JSON data back to the output file
    # save_json_data(output_file, existing_data)
    # print(f"Search results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some search results.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--seed_queries_file', type=str, required=True, help='Path to the seed queries file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')

    args = parser.parse_args()
    with open(args.seed_queries_file, 'r') as file:
        seed_queries = file.read().splitlines()

    for query in seed_queries:
        process_search_results(args.file_path, query, args.output_file)
