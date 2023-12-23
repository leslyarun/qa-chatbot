# prompt_template.py

"""
This file contains the template for generating prompts for the document assistant.

The prompts are used to guide the interaction between the system and the human in order to generate accurate answers based on the given context and questions.

The template includes three types of prompts:
- SystemMessagePromptTemplate: Represents a system message prompt that provides information or instructions to the human.
- HumanMessagePromptTemplate: Represents a human message prompt that includes the context and question for generating an answer.
- ChatPromptTemplate: Represents a chat prompt that combines the system and human message prompts to facilitate the conversation between the system and the human.

"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

TEMPLATE = (
    "You're a PDF answer extractor. You help the user to answer questions from a PDF."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)

HUMAN_TEMPLATE = """

I have extracted text from a PDF, which includes specific context and a related question. I require an accurate answer based on this context. Please adhere to the following guidelines:

1. Carefully read the provided context and the question.
2. Provide an answer derived solely from the given context.
3. If the context contains the exact answer, replicate it verbatim.
4. Compute confidence score for every answer.
4. If the confidence score is low, simply respond with "Data Not Available". Avoid elaborating or guessing.

Context:
{context}

Question:
{question}

Please provide the answer as plain text, focusing on precision and adherence to these instructions.

"""
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
