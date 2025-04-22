import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from openai import AzureOpenAI

# .env を読み込み
load_dotenv()

# サポートする埋め込みモデル
EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

def _validate_model(model):
    if model not in EMBEDDING_MODELS:
        raise RuntimeError(f"Invalid embedding model: {model}, available models: {EMBEDDING_MODELS}")

def request_to_embed(texts, model):
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"
    if use_azure:
        return request_to_azure_embed(texts, model)
    _validate_model(model)
    return OpenAIEmbeddings(model=model).embed_documents(texts)

def request_to_azure_embed(texts, model):
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_EMBEDDING_VERSION"),
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
    )
    response = client.embeddings.create(input=texts, model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"))
    return [item.embedding for item in response.data]
