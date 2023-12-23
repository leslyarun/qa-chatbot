# constants.py

"""
This module contains the constants used in the application.
"""

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# The API key for the OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
