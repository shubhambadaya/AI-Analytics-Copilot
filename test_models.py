import google.generativeai as genai
import os
from src.utils.config import config

genai.configure(api_key=config.GEMINI_API_KEY)
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
