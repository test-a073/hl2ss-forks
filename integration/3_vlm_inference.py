import random
import time

def policy_function() -> int:
    """Returns the device for inference.
    0 : Workstation
    1 : Jetson Nano
    """
    return random.randint(0, 1)

import json

# Specify the JSON file path
input_file = "sasika_stream/queries.json"

try:
    # Load the JSON file into the dictionary
    with open(input_file, "r") as json_file:
        queries = json.load(json_file)
    print("Data successfully loaded into 'queries':", queries)
except FileNotFoundError:
    print(f"File {input_file} not found.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")


image_path = queries["image"]
query_text = queries["text"]

# Determine the inference device
inference_device = policy_function()

if inference_device == 0:
    import requests

    server_url = "http://10.4.16.46:5000/query"
    image_path = image_path 
    query_text = query_text

    try:
        start_time = time.time()

        # Prepare the request payload
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'query': query_text}

            # Send the POST request
            response = requests.post(server_url, files=files, data=data)

        # Measure processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Handle the response
        if response.status_code == 200:
            print("Response:", response.json().get('response', "No response field"))
        else:
            print("Error:", response.json())

        print(f"Processing Time: {processing_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")

elif inference_device == 1:
    import ollama

    try:
        start_time = time.time()

        # Perform inference using Jetson Nano with the Moondream model
        jetson_ollama_response = ollama.chat(
            model="moondream",
            messages=[{
                "role": "user",
                "content": query_text,
                "images": [image_path]
            }]
        )

        # Measure processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Extract and structure the response
        jetson_ollama_response_dict = dict(jetson_ollama_response)
        jetson_ollama_message_content = jetson_ollama_response_dict.get('message', {}).get('content', "No content")

        response = {
            "model": jetson_ollama_response_dict.get('model', "Unknown model"),
            "message": jetson_ollama_message_content
        }

        print("Response:", response)
        print(f"Processing Time: {processing_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Invalid inference device.")
