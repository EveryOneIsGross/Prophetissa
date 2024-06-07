# v 02
# to use from teh commandline 
# python judgeDREAD.py path_to_your_input_json_file.json --threshold 0.75

import json
import datetime
import argparse

def parse_context_to_messages(context):
    """
    Parse the context string into a list of messages, each with a relevance score and sentiment.
    """
    chunks = context.split("\n\n")
    messages = []
    for chunk in chunks:
        if chunk:
            parts = chunk.split("\n")
            content = parts[0].replace("Chunk: ", "")
            relevance = float(parts[1].replace("Relevance Score: ", ""))
            sentiment = float(parts[2].replace("Sentiment: ", ""))
            messages.append((content, relevance, sentiment))
    return messages

def calculate_weight(relevance, sentiment):
    """
    Calculate a weight for the message based on relevance and sentiment.
    """
    return relevance * (1 + sentiment)

def format_query_with_context(query, context):
    """
    Format the query to include the context information as specified.
    """
    return f"""
Context information is below
---------------------
{context}
---------------------
Given the context information and not prior knowledge,
answer the query.
Query: {query}
"""

def filter_and_convert_messages(data, threshold):
    """
    Filter out messages below a certain weight threshold and convert others into the instruct format.
    """
    accepted_data = []
    rejected_data = []
    for item in data:
        messages = []
        context_messages = parse_context_to_messages(item['context'])
        full_context = "\n\n".join([f"Chunk: {content}\nRelevance Score: {relevance}\nSentiment: {sentiment}" for content, relevance, sentiment in context_messages])
        formatted_query = format_query_with_context(item['question'], full_context)
        for content, relevance, sentiment in context_messages:
            weight = calculate_weight(relevance, sentiment)
            if weight < threshold:
                rejected_data.append({"query": formatted_query, "context": full_context, "question": item['question'], "answer": item['answer']})
            else:
                messages.append({"role": "user", "content": formatted_query})
                messages.append({"role": "assistant", "content": item['answer']})
        if messages:
            accepted_data.append({"messages": messages})
    return accepted_data, rejected_data

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_jsonl_data(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def save_json_data(data, file_path):
    with open(file_path, 'w') as file:
        file.write(json.dumps(data, indent=4))
        file.write('\n')

def main(input_file, threshold):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = timestamp + input_file.replace('.json', '_pass.jsonl')
    rejected_file = timestamp + input_file.replace('.json', '_fail.jsonl')

    data = load_json_data(input_file)
    accepted_data, rejected_data = filter_and_convert_messages(data, threshold)

    # Save the data that meets the threshold
    save_jsonl_data(accepted_data, output_file)
    save_json_data(accepted_data, timestamp + input_file.replace('.json', '_pass.json'))
    print(f"Data that meets the threshold is converted and saved to {output_file}")

    # Save the data that does not meet the threshold
    save_jsonl_data(rejected_data, rejected_file)
    save_json_data(rejected_data, timestamp + input_file.replace('.json', '_fail.json'))
    print(f"Data below the threshold is saved to {rejected_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input JSON for threshold filtering.")
    parser.add_argument("input_file", type=str, help="The path to the input JSON file.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Weight threshold for filtering entries (default: 0.75).")
    args = parser.parse_args()
    main(args.input_file, args.threshold)