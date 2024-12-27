from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8004/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
