#vLLM

# vLLM is easy to install and use, with both OpenAI API compatibility
# and a native Pythoninterface.

# First, launch the vLLM OpenAI-compatible server:
"""
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --host 0.0.0.0 \
    --port 8000
"""

from huggingface_hub import InferenceClient

# Initialize client pointing to vLLM endpoint
client = InferenceClient(
    model="http://localhost:8000/v1",  # URL to the vLLM server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Alternatively, you can use the OpenAI client:

from openai import OpenAI

# Initialize client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# more exemples: https://github.com/huggingface/course/blob/main/chapters/en/chapter2/8.mdx