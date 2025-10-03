# verify_weather_api.py
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
CITY = "Mumbai"

if not API_KEY:
    print("❌ ERROR: OPENWEATHERMAP_API_KEY not found in your .env file. Please check the variable name.")
else:
    print(f"Testing API key ending in '...{API_KEY[-4:]}' for city '{CITY}'...")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        status_code = response.status_code

        print(f"--> Request URL: {url}")
        print(f"--> Response Status Code: {status_code}")
        print("\n--- RAW RESPONSE ---")
        print(response.json())
        print("----------------------\n")

        if status_code == 200:
            print("✅ SUCCESS: The API key is active and the request was successful.")
        elif status_code == 401:
            print("❌ FAILURE: Status code 401 indicates an INVALID or INACTIVE API key.")
            print("Please double-check the key in your .env file. If it's correct, you must wait longer for it to activate.")
        else:
            print(f"⚠️ UNKNOWN ERROR: The API returned status code {status_code}. The key might be active but there could be another issue.")

    except Exception as e:
        print(f"A network or request error occurred: {e}")
