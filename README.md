# Zania.AI QA App

## Overview

This AI Query Application is designed to process, embed, and answer queries based on textual data extracted from PDF documents. Utilizing the LangChain framework along with OpenAI's Chat Models, this application offers a sophisticated solution for extracting and interpreting information from unstructured text. It's primarily built to serve as a backend for systems requiring AI-based text processing and query answering functionalities.

## Key Features

- **Document Processing**: Load and process PDF documents, splitting them into manageable text chunks.
- **Text Embedding**: Utilize HuggingFace's transformer models to embed text data.
- **Query Answering**: Leverage OpenAI's Chat Models to generate relevant answers based on the processed text.

## Dependencies

To run this application, you'll need the following:
- LangChain (for document loading, text splitting, and embedding)
- HuggingFace's Transformers (for embeddings)
- OpenAI's GPT models (for the chat model)
- LangChain Core (for runnables and output parsers)
- FastAPI and Uvicorn (for web API functionalities)

## Installation

Before running the application, ensure you have all the dependencies installed. You can install these using pip:

```bash
pip install -r requirements.txt
```

## Usage

**How to run?**
```
cp .env.sample .env
    update your open_ai_key in .env

python main.py
```

## Making Queries
Send a POST request to `http://localhost:8000/query/` with a PDF document and a list of questions. The application will return the AI-generated answers.

## Sample Q&A screenshot from Swagger

<img width="1299" alt="image" src="https://github.com/leslyarun/zaniaqa/assets/5101854/fa8917f3-b20f-48b1-bfb5-2f7d939a785d">

<img width="1418" alt="image" src="https://github.com/leslyarun/zaniaqa/assets/5101854/0bc1944a-a0f0-4ff3-b89a-61cc03cd88fb">

## Using Postman

<img width="1082" alt="image" src="https://github.com/leslyarun/zaniaqa/assets/5101854/638009b2-6ae0-42f9-84a2-88ca4a415bf9">

## Using CURL

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'questions=What is the name of the company?,Who is the CEO of the company?,What is their vacation policy?,What is the termination policy?' \
  -F 'document=@handbook.pdf;type=application/pdf'
```

## Author

- Lesly
- Date: 2023-06-23
