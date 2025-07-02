import os
from dotenv import load_dotenv

# Load the .env file explicitly
load_dotenv()

print("Environment Variables Test:")
print("-" * 50)
print("Raw environment variables:")
for key in ["ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY_2", "ELEVENLABS_API_KEY_3", "ELEVENLABS_API_KEY_4"]:
    value = os.getenv(key)
    print(f"{key}: {'Present' if value else 'Not found'}")
    if value:
        print(f"Value: {value}")

print("\nBoolean settings (raw values):")
for key in ["TEST_MODE_ENABLED", "DEVELOPMENT_SKIP_MODE", "CREDIT_PROTECTION_ENABLED"]:
    print(f"{key}: {os.getenv(key)}")
print("-" * 50) 