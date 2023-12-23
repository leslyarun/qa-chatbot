# utils.py

"""
Text Processing and Query Answering Module using LangChain and OpenAI's Chat Models

This module provides a set of functions to load, process, and embed textual data from unstructured PDF documents.
It uses LangChain's capabilities for document loading, text splitting, and embedding, along with OpenAI's chat models for generating answers to queries based on the processed text.

Functions:
    - data_load_chunk(filepath): Loads a PDF document, splits it into chunks, and returns a list of document chunks.
    - embedd(docs): Creates embeddings for a list of documents using HuggingFaceEmbeddings and stores them in a Chroma vectorstore.
    - chunk_embedd(filepath): A combination function that processes a PDF file and returns its embedded vectorstore.
    - question_answer(filepath, query): Generates answers to queries using the ChatOpenAI model, based on the context provided by the vectorstore created from the PDF file.

Dependencies:
    - LangChain (for document loading, text splitting, and embedding)
    - HuggingFace's Transformers (for embeddings)
    - OpenAI's GPT models (for the chat model)
    - LangChain Core (for runnables and output parsers)

Usage:
The module is designed to be used as part of a larger system that requires processing text from PDFs and generating AI-based answers. It can be imported and its functions can be called with the necessary arguments.

Example:
To get answers for a query based on a PDF document:

docs = data_load_chunk("example.pdf")
vectorstore = embedd(docs)
answer = question_answer("example.pdf", "What is the main topic of the document?")


Author: Lesly
Date: 2023-06-23
"""

from typing import List
import tiktoken
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from prompt_template import chat_prompt


def data_load_chunk(filepath: str):
    """
    Load and split a document into chunks of text.

    Args:
        filepath (str): The path to the document file.

    Returns:
        List[Document]: A list of document chunks.
    """
    loader = UnstructuredPDFLoader(filepath)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
        length_function=length_function,
    )
    docs = text_splitter.split_documents(doc)
    return docs


def embedd(docs: List[str]):
    """
    Embeds a list of documents using a pre-trained model.

    Args:
        docs (List[str]): The list of documents to embed.

    Returns:
        Chroma: The vectorstore containing the embeddings.
    """
    model_name = "TaylorAI/gte-tiny"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma.from_documents(docs, embeddings)  # type: no-check

    return vectorstore


def chunk_embedd(filepath: str):
    """
    Embeds the documents in the given file and returns a vector store.

    Args:
        filepath (str): The path to the file containing the documents.

    Returns:
        List[float]: The vector store containing the embedded documents.
    """
    docs = data_load_chunk(filepath)
    vectorstore = embedd(docs)
    return vectorstore


def length_function(x):
    """
    Calculates the length of the encoded representation of x.

    Args:
        x: The input to be encoded and measured.

    Returns:
        int: The length of the encoded representation.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(str(x)))


async def question_answer(filepath, query) -> str:
    """
    Given a filepath and a query, this function generates an answer using a ChatOpenAI model.

    Args:
        filepath (str): The path to the file containing the vector store.
        query (str): The query string used to generate the answer.

    Returns:
        str: The generated answer.
    """
    # Initialize the ChatOpenAI model
    model = ChatOpenAI()

    # Load the vectorstore from the given filepath
    vectorstore = chunk_embedd(filepath)

    # Create a retriever using the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    # Set up the pipeline for generating the answer
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | chat_prompt | model | StrOutputParser()

    # Generate the answer using the pipeline
    answer = await chain.ainvoke(query)

    # Return the generated answer
    return answer
