# prompt_template.py


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = (
    "You're a document assistant. You help the user to answer questions from a PDF."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = """

I have extracted text from a PDF, which includes specific context and a related question. I require an accurate answer based on this context. Please adhere to the following guidelines:

1. Carefully read the provided context and the question.
2. Provide an answer derived solely from the given context.
3. If the context contains the exact answer, replicate it verbatim.
4. Compute confidence score for every answer.
4. If the confidence score is below 0.15, simply respond with "Data Not Available". Avoid elaborating or guessing.

Context:
{context}

Question:
{question}

Please provide the answer as plain text, focusing on precision and adherence to these instructions.

"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
