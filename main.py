# main.py

"""_summary_

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""

import os
import shutil
from typing import List
import openai
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File

from utils import question_answer
from constants import OPENAI_API_KEY

if OPENAI_API_KEY is None:
    # Handle the case when the environment variable is not set
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

app = FastAPI()


@app.post("/query/")
async def generate_text(
    questions: List[str] = Form(...), document: UploadFile = File(...)
):
    # Extract the first item from the list and split it into separate questions
    if questions and len(questions) == 1:
        questions_list = [question.strip() for question in questions[0].split(",")]
    else:
        questions_list = questions

    temp_file_path = f"temp_{document.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(document.file, buffer)

    qs_answers = {}
    for question in questions_list:
        answer = await question_answer(temp_file_path, question)
        qs_answers[question] = answer

    # Clean up: remove the temporary file
    os.remove(temp_file_path)

    return qs_answers


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
