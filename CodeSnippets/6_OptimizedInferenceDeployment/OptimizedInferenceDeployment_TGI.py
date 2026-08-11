# TGI

# First, launch the TGI server using Docker:
# (Não roda no MAC, precisa ser linux)
"""
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
"""

from huggingface_hub import InferenceClient

# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
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

# Initialize client pointing to TGI endpoint
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Make sure to include /v1
    api_key="not-needed",  # TGI doesn't require an API key by default
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