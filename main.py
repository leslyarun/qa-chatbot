# main.py

"""
AI Query Application using LLMs

This script serves as the main entry point for a FastAPI application designed to answer questions based on the content of a provided document. It utilizes OpenAI's GPT 3.5 models to process natural language queries.

The application allows users to upload a document and submit a list of questions. Each question is then answered by the AI model using the content of the uploaded document as context. 

The script sets up the FastAPI application, configures the necessary environment variables, and defines the endpoint for processing the queries. It also includes error handling for missing API keys and file management for handling the uploaded documents.

Key Components:
- FastAPI setup for handling HTTP requests
- OpenAI GPT integration for generating answers
- File handling for uploaded documents
- Error handling for environment configuration

Usage:
Run the script using a WSGI server like Uvicorn to start the FastAPI application. Make POST requests to the specified endpoint with the required data.

Dependencies:
- FastAPI
- Uvicorn
- OpenAI

Example:
To start the server, run: `uvicorn main:app --reload`. Then, make POST requests to `http://localhost:8000/query/`.

Author: Lesly 
Date: 2023-06-23
"""


import os
import shutil
from typing import List, Dict
import openai
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File

from utils import question_answer
from constants import OPENAI_API_KEY

# Check if OPENAI_API_KEY environment variable is set, and raise an error if not
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Set the environment variable and OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize the FastAPI app
app = FastAPI()


@app.post("/query/")
async def generate_text(
    questions: List[str] = Form(...), document: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Process POST request to generate text based on provided questions and a document.

    Args:
        questions (List[str]): A list of questions provided by the user.
        document (UploadFile): An uploaded file that contains the context for answering the questions.

    Returns:
        Dict[str, str]: A dictionary where keys are questions and values are corresponding answers.
    """
    # Process questions
    if questions and len(questions) == 1:
        questions_list = [question.strip() for question in questions[0].split(",")]
    else:
        questions_list = questions

    # Write the uploaded file to a temporary path
    temp_file_path = f"temp_{document.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(document.file, buffer)

    # Generate answers for each question
    qs_answers = {}
    for question in questions_list:
        answer = await question_answer(temp_file_path, question)
        qs_answers[question] = answer

    # Clean up by removing the temporary file
    os.remove(temp_file_path)

    return qs_answers


# Run the application with Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
