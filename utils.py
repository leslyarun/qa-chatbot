from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from prompt_template import chat_prompt


def data_load_chunk(filepath):
    loader = UnstructuredPDFLoader(filepath)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    docs = text_splitter.split_documents(doc)
    return docs


def embedd(docs):
    model_name = "TaylorAI/gte-tiny"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma.from_documents(docs, embeddings)  # type: no-check

    return vectorstore


def chunk_embedd(filepath):
    docs = data_load_chunk(filepath)
    vectorstore = embedd(docs)
    return vectorstore


async def question_answer(filepath, query) -> str:
    model = ChatOpenAI()
    vectorstore = chunk_embedd(filepath)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    print(retriever.get_relevant_documents(query))

    # Generate the answer given the context
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | chat_prompt | model | StrOutputParser()

    answer = await chain.ainvoke(query)
    return answer
