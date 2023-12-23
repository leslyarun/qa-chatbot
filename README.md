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

**Making Queries**: Send a POST request to `http://localhost:8000/query/` with a PDF document and a list of questions. The application will return the AI-generated answers.

## Author

- Lesly
- Date: 2023-06-23