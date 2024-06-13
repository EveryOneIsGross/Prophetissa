# v 09
# to use from the command line
# python judgeDREAD.py path_to_your_input_json_file.json --threshold 0.75
# still a bit broken do to the new keys in the json from the new search types

import json
import datetime
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from openai import OpenAI
import re

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

def parse_context_to_messages(context):
    messages = []
    lines = context.split('[1, N/A]\n\n')
    for line in lines:
        messages.append((line.strip(), 0.0))  # Add the context message with a default relevance score
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

def judge_answer(item):
    """
    Use the LLM to judge whether the answer adequately addresses the question based on the provided context, query, and available fields.
    """
    query = item['query']
    question = item['question']
    answer = item['answer']
    context = item['context']

    prompt = f"""
    Domain/Seed Term (Query): {query}
    Specific Question: {question}
    Answer: {answer}
    Context:
    {context}
    """

    if 'chunks' in item and item['chunks']:
        if 'image_filenames' in item['chunks'][0] and 'image_descriptions' in item['chunks'][0]:
            image_filenames = ', '.join(item['chunks'][0]['image_filenames']) if item['chunks'][0]['image_filenames'] else 'N/A'
            image_descriptions = ', '.join(item['chunks'][0]['image_descriptions']) if item['chunks'][0]['image_descriptions'] else 'N/A'
            prompt += f"Image Filenames: {image_filenames}\nImage Descriptions: {image_descriptions}\n"

    prompt += """
    Evaluate whether the provided answer adequately addresses the specific question, considering the overall domain provided by the query and the detailed context. Provide a floating point score between 0 and 1, where 1 indicates a highly relevant and satisfactory answer, and 0 indicates an irrelevant or inadequate answer.
    """

    response = client.chat.completions.create(
        model="qwen2:0.5b-instruct-fp16",
        messages=[
            {"role": "system", "content": "You are to assess the relevance and adequacy of the answer given the context, the query as the general topic, and the specific question."},
            {"role": "user", "content": prompt},
        ]
    )

    response_text = response.choices[0].message.content.strip()
    match = re.search(r"([-+]?\d*\.\d+|\d+)", response_text)
    if match:
        judge_score = float(match.group())
    else:
        judge_score = 0.0  # Default value if no numeric score is found

    print(f"Prompt: {prompt}")
    print(f"Response: {response_text}")
    print(f"Judge Score: {judge_score}")
    return judge_score

def filter_and_convert_messages(data, threshold):
    """
    Filter out messages below a certain weight threshold and convert others into the instruct format.
    """
    accepted_data = []
    rejected_data = []
    seen_entries = set()  # Set to keep track of unique entries

    for item in data:
        context_messages = parse_context_to_messages(item['context'])
        full_context = "\n\n".join([f"Chunk: {content}\nRelevance Score: {relevance}" for content, relevance in context_messages])
        formatted_query = format_query_with_context(item['question'], full_context)
        entry_key = (item['question'], item['answer'], full_context)  # Create a unique key for each entry

        if entry_key in seen_entries:
            # Skip duplicate entries
            continue

        seen_entries.add(entry_key)  # Add the entry key to the set of seen entries

        judge_score = judge_answer(item)

        chunks = []
        for content, relevance in context_messages:
            weight = calculate_weight(relevance, 0.0, judge_score)  # Assuming default sentiment as 0.0
            if weight >= threshold:
                chunks.append({
                    "text": content,
                    "relevance_score": relevance,
                    "sentiment": 0.0  # Assuming default sentiment as 0.0
                })
        if chunks:
            accepted_data.append({
                "query": item['query'],
                "context": full_context,
                "chunks": chunks,
                "combined_score": item.get('combined_score', 0.0),  # Provide a default value of 0.0 if 'combined_score' is missing
                "question": item['question'],
                "answer": item['answer'],
                "judge_score": judge_score
            })
        else:
            rejected_data.append({
                "query": item['query'],
                "context": full_context,
                "chunks": chunks,
                "combined_score": item.get('combined_score', 0.0),  # Provide a default value of 0.0 if 'combined_score' is missing
                "question": item['question'],
                "answer": item['answer'],
                "judge_score": judge_score
            })

    return accepted_data, rejected_data

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_xml_data(data, file_path):
    root = ET.Element('dataset')
    for entry in data:
        item = ET.SubElement(root, 'item')
        for key, value in entry.items():
            if key == 'chunks':
                chunks = ET.SubElement(item, 'chunks')
                for chunk in value:
                    chunk_elem = ET.SubElement(chunks, 'chunk')
                    for chunk_key, chunk_value in chunk.items():
                        chunk_child = ET.SubElement(chunk_elem, chunk_key)
                        chunk_child.text = str(chunk_value)
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
            if child.tag == 'chunks':
                chunks = []
                for chunk in child.findall('chunk'):
                    chunk_data = {}
                    for chunk_child in chunk:
                        chunk_data[chunk_child.tag] = chunk_child.text
                    chunks.append(chunk_data)
                entry['chunks'] = chunks
            else:
                entry[child.tag] = child.text
        data.append(entry)
    return data

def save_jsonl_data(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            jsonl_entry = {
                "query": entry["query"],
                "context": entry["context"],
                "chunks": entry["chunks"],
                "combined_score": entry["combined_score"],
                "question": entry["question"],
                "answer": entry["answer"]
            }
            json.dump(jsonl_entry, file)
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
    weights = [item['combined_score'] for item in accepted_data]
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
    high_quality_data = [item for item in accepted_data if item['combined_score'] >= adjusted_threshold]
    medium_quality_data = [item for item in accepted_data if adjusted_threshold > item['combined_score'] >= threshold]
    low_quality_data = [item for item in accepted_data if item['combined_score'] < threshold]

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

