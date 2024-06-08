# v 05
# to use from the command line
# python judgeDREAD.py path_to_your_input_json_file.json --threshold 0.75

import json
import datetime
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

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

def calculate_weight(relevance, sentiment, judge_score):
    """
    Calculate a weight for the message based on relevance, sentiment, and judge score.
    """
    return relevance * (1 + sentiment) * judge_score

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

import re

def judge_answer(question, answer, context, combined_score):
    """
    Use the LLM to judge whether the answer adequately addresses the question based on the provided context and combined score.
    """
    prompt = f"""
Question: {question}
Answer: {answer}
Context:
{context}
Combined Score: {combined_score}

Given the context information, its relevance and sentiment scores, and the combined score, evaluate whether the provided answer adequately addresses the question.
Provide a floating point score between 0 and 1, where 1 indicates a highly relevant and satisfactory answer, and 0 indicates an irrelevant or inadequate answer.

Example : 1 = good, 0.8 = good enough, 0.5 = neutral, 0 = bad

Judge Score:"""

    response = client.chat.completions.create(
        model="qwen2:0.5b-instruct-fp16",
        messages=[
            {"role": "system", "content": "You are a lazy apathetic assistant, do the bare minimum when evaluating the quality of answers based on the provided context."},
            {"role": "user", "content": prompt},
        ]
    )

    response_text = response.choices[0].message.content.strip()
    match = re.search(r"([-+]?\d*\.\d+|\d+)", response_text)
    if match:
        judge_score = float(match.group())
    else:
        judge_score = 0.0  # Default value if no numeric score is found

    print(f"{prompt}")
    print(f"{response_text}")
    print(f"{judge_score}")
    return judge_score

def filter_and_convert_messages(data, threshold):
    """
    Filter out messages below a certain weight threshold and convert others into the instruct format.
    """
    accepted_data = []
    rejected_data = []
    seen_entries = set()  # Set to keep track of unique entries

    for item in data:
        messages = []
        context_messages = parse_context_to_messages(item['context'])
        full_context = "\n\n".join([f"Chunk: {content}\nRelevance Score: {relevance}\nSentiment: {sentiment}" for content, relevance, sentiment in context_messages])
        formatted_query = format_query_with_context(item['question'], full_context)
        entry_key = (item['question'], item['answer'], full_context)  # Create a unique key for each entry

        if entry_key in seen_entries:
            # Skip duplicate entries
            continue

        seen_entries.add(entry_key)  # Add the entry key to the set of seen entries

        judge_score = judge_answer(item['question'], item['answer'], full_context, item['combined_score'])
        chunks = []
        for index, (content, relevance, sentiment) in enumerate(context_messages, start=1):
            weight = calculate_weight(relevance, sentiment, judge_score)
            if weight >= threshold:
                chunks.append({
                    "index": index,
                    "text": content,
                    "relevance_score": relevance,
                    "sentiment": sentiment
                })
                messages.append({"role": "user", "content": formatted_query})
                messages.append({"role": "assistant", "content": item['answer']})
        if messages:
            accepted_data.append({"messages": messages, "judge_score": judge_score, "context": full_context, "chunks": chunks})
        else:
            rejected_data.append({"query": formatted_query, "context": full_context, "question": item['question'], "answer": item['answer'], "judge_score": judge_score})

    return accepted_data, rejected_data

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_xml_data(data, file_path):
    root = ET.Element('dataset')
    for entry in data:
        item = ET.SubElement(root, 'item')
        for key, value in entry.items():
            if key == 'messages':
                messages = ET.SubElement(item, 'messages')
                for message in value:
                    message_elem = ET.SubElement(messages, 'message')
                    role = ET.SubElement(message_elem, 'role')
                    role.text = message['role']
                    content = ET.SubElement(message_elem, 'content')
                    content.text = message['content']
            elif key == 'chunks':
                chunks = ET.SubElement(item, 'chunks')
                for chunk in value:
                    chunk_elem = ET.SubElement(chunks, 'chunk')
                    chunk_elem.set('index', str(chunk['index']))
                    text = ET.SubElement(chunk_elem, 'text')
                    text.text = chunk['text']
                    relevance_score = ET.SubElement(chunk_elem, 'relevance_score')
                    relevance_score.text = str(chunk['relevance_score'])
                    sentiment = ET.SubElement(chunk_elem, 'sentiment')
                    sentiment.text = str(chunk['sentiment'])
            else:
                elem = ET.SubElement(item, key)
                elem.text = str(value)

    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(file_path, 'w') as file:
        file.write(xml_string)

def load_xml_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for item in root.findall('item'):
        entry = {}
        for child in item:
            if child.tag == 'messages':
                messages = []
                for message in child.findall('message'):
                    message_data = {}
                    for message_child in message:
                        message_data[message_child.tag] = message_child.text
                    messages.append(message_data)
                entry['messages'] = messages
            elif child.tag == 'chunks':
                chunks = []
                for chunk in child.findall('chunk'):
                    chunk_data = {
                        'index': int(chunk.get('index')),
                        'text': chunk.find('text').text,
                        'relevance_score': float(chunk.find('relevance_score').text),
                        'sentiment': float(chunk.find('sentiment').text)
                    }
                    chunks.append(chunk_data)
                entry['chunks'] = chunks
            else:
                entry[child.tag] = child.text
        data.append(entry)
    return data

def save_jsonl_data(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def save_json_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def main(input_file, threshold):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_xml = timestamp + input_file.replace('.json', '_pass.xml')
    output_file_jsonl = timestamp + input_file.replace('.json', '_pass.jsonl')
    output_file_json = timestamp + input_file.replace('.json', '_pass.json')
    rejected_file_xml = timestamp + input_file.replace('.json', '_fail.xml')
    rejected_file_jsonl = timestamp + input_file.replace('.json', '_fail.jsonl')
    rejected_file_json = timestamp + input_file.replace('.json', '_fail.json')

    data = load_json_data(input_file)
    accepted_data, rejected_data = filter_and_convert_messages(data, threshold)

    # Analyze the distribution of weights
    weights = [item['judge_score'] for item in accepted_data]
    mean_weight = sum(weights) / len(weights)
    median_weight = sorted(weights)[len(weights) // 2]
    std_dev_weight = (sum((w - mean_weight) ** 2 for w in weights) / len(weights)) ** 0.5

    print(f"Weight Distribution Analysis:")
    print(f"Mean Weight: {mean_weight:.2f}")
    print(f"Median Weight: {median_weight:.2f}")
    print(f"Standard Deviation: {std_dev_weight:.2f}")

    # Adjust the threshold dynamically
    adjusted_threshold = mean_weight + std_dev_weight
    print(f"Adjusted Threshold: {adjusted_threshold:.2f}")

    # Categorize the data based on multiple thresholds
    high_quality_data = [item for item in accepted_data if item['judge_score'] >= adjusted_threshold]
    medium_quality_data = [item for item in accepted_data if adjusted_threshold > item['judge_score'] >= threshold]
    low_quality_data = [item for item in accepted_data if item['judge_score'] < threshold]

    # Save the categorized data as XML, JSONL, and JSON files
    save_xml_data(high_quality_data, timestamp + input_file.replace('.json', '_high_quality.xml'))
    save_jsonl_data(high_quality_data, timestamp + input_file.replace('.json', '_high_quality.jsonl'))
    save_json_data(high_quality_data, timestamp + input_file.replace('.json', '_high_quality.json'))

    save_xml_data(medium_quality_data, timestamp + input_file.replace('.json', '_medium_quality.xml'))
    save_jsonl_data(medium_quality_data, timestamp + input_file.replace('.json', '_medium_quality.jsonl'))
    save_json_data(medium_quality_data, timestamp + input_file.replace('.json', '_medium_quality.json'))

    save_xml_data(low_quality_data, timestamp + input_file.replace('.json', '_low_quality.xml'))
    save_jsonl_data(low_quality_data, timestamp + input_file.replace('.json', '_low_quality.jsonl'))
    save_json_data(low_quality_data, timestamp + input_file.replace('.json', '_low_quality.json'))

    # Save the accepted and rejected data as XML, JSONL, and JSON files
    save_xml_data(accepted_data, output_file_xml)
    save_jsonl_data(accepted_data, output_file_jsonl)
    save_json_data(accepted_data, output_file_json)

    save_xml_data(rejected_data, rejected_file_xml)
    save_jsonl_data(rejected_data, rejected_file_jsonl)
    save_json_data(rejected_data, rejected_file_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input JSON for threshold filtering.")
    parser.add_argument("input_file", type=str, help="The path to the input JSON file.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Weight threshold for filtering entries (default: 0.75).")
    args = parser.parse_args()
    main(args.input_file, args.threshold)
