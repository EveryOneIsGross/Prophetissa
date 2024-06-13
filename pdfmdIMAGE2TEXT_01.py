import os
import re
import ollama

def process_markdown_file(file_path, image_folder):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    def replace_link(match):
        image_filename = match.group(1)
        image_path = os.path.join(image_folder, image_filename)

        if os.path.exists(image_path):
            max_retries = 5
            for attempt in range(max_retries):
                res = ollama.chat(
                    model="moondream",
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Describe this black and white graph, note relevant shapes, features and words:',
                            'images': [image_path]
                        }
                    ]
                )
                image_description = res['message']['content'].strip()
                image_description = ' '.join(image_description.split())

                if image_description:
                    print(f"Generated description for {image_filename}: {image_description}")
                    return f"![{image_description}]({image_filename})"
                else:
                    print(f"Empty response for {image_filename}, attempt {attempt + 1} of {max_retries}")

            print(f"Failed to generate description for {image_filename} after {max_retries} attempts.")
            return match.group(0)
        else:
            return match.group(0)

    print("Processing file:", file_path)
    updated_content = re.sub(r'!\[.*?\]\((.*?)\)', replace_link, content)
    updated_content = updated_content.strip()

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                process_markdown_file(file_path, folder_path)

if __name__ == '__main__':
    folder_path = 'WALTER_RUSSEL\WalterRusselbooks\Walter Russell - books\Home Study Course\Home Study Course Unit 12 Lessons 45,46,47,48 by Walter Russell'
    process_folder(folder_path)
