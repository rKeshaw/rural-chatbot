# list_models.py - A script to get the current list of available models from Groq.
import os
from groq import Groq
from dotenv import load_dotenv

# Load the .env file to get the API key
load_dotenv()

try:
    # Initialize the Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print("Fetching available models from Groq...\n")

    # Get the list of models
    models = client.models.list().data

    # Find the longest model ID for clean printing
    max_id_length = max(len(model.id) for model in models)

    print(f"{'Model ID'.ljust(max_id_length)} | {'Active'.ljust(8)} | {'Owned By'}")
    print(f"{'-' * max_id_length} | {'-' * 8} | {'-' * 15}")

    for model in models:
        model_id = model.id.ljust(max_id_length)
        active = str(model.active).ljust(8)
        owned_by = model.owned_by
        print(f"{model_id} | {active} | {owned_by}")

except Exception as e:
    print(f"An error occurred: {e}")
