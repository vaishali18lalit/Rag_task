import uuid
# from langchain.vectorstores import Chroma
import tiktoken
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
import json
from base64 import b64decode
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def truncate_text_by_tokens(text, max_tokens=127500):
    """Truncate a text string by token count safely."""
    tokenized = tokenizer.encode(text)  # Convert text to tokens
    truncated_text = tokenizer.decode(tokenized[:max_tokens])  # Truncate and decode back
    return truncated_text

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(str(doc))
            print("hellllooo",text)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for i, text_element in enumerate(docs_by_type["texts"], start=1):  # Start chunk numbering from 1
            context_text += f"Chunk {i}: {text_element}\n\n"

    # construct prompt with context (including images)
    # Construct prompt with context
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
    At the end of your response, include relevant chunk numbers and image numbers separately in the format and put {None} if not relevant to the question.:
    YOU MUST follow the format : Relevant Chunks: {{chunk_numbers}} and Relevant Images: {{image_numbers}}
    Context:
    {context_text}
    Question: {user_question}
    """
    truncated_prompt = truncate_text_by_tokens(prompt_template, max_tokens=126500)
    prompt_content = [{"type": "text", "text": truncated_prompt}]
    if len(docs_by_type["images"]) > 0:
        for i, image in enumerate(docs_by_type["images"], start=1):
            prompt_content.append(f"Image {i}")
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def initialize_retriever_for_document(collection_name):
    """Initialize a separate vector store for each document."""
    vectorstore = Chroma(collection_name=collection_name, embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    retriever.search_kwargs['k'] = 4  # Setting search parameter
    return retriever

def generate_ids(data):
    """Generate unique IDs for given data."""
    return [str(uuid.uuid4()) for _ in data]

def add_documents_to_retriever(retriever, data, summaries, doc_name):
    """Add documents and summaries to the retriever."""
    doc_ids = generate_ids(data)
    summary_docs = [
        Document(page_content=summary, metadata={"doc_id": doc_ids[i], "doc_name": doc_name})
        for i, summary in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, data)))

# **New function to load data once globally**
def load_data_and_initialize_retrievers():
    """Loads data and initializes retrievers only once when the app starts."""
    retriever_doc_1 = initialize_retriever_for_document("doc_1_collection")
    retriever_doc_2 = initialize_retriever_for_document("doc_2_collection")

    # Load JSON files once
    with open("json_files/chunks_2023.json", "r") as file:
        data = json.load(file)
    texts_2023, tables_2023, images_2023 = data["text_chunks"], data["table_chunks"], data["image_chunks"]

    with open("json_files/chunks_2024.json", "r") as file:
        data = json.load(file)
    texts_2024, tables_2024, images_2024 = data["text_chunks"], data["table_chunks"], data["image_chunks"]

    with open("json_files/summaries_2023.json", "r") as file:
        data = json.load(file)
    text_summaries_2023, table_summaries_2023, image_summaries_2023 = (
        data["text_summaries"], data["table_summaries"], data["image_summaries"]
    )

    with open("json_files/summaries_2024.json", "r") as file:
        data = json.load(file)
    text_summaries_2024, table_summaries_2024, image_summaries_2024 = (
        data["text_summaries"], data["table_summaries"], data["image_summaries"]
    )

    # Add text, table, and image summaries to the retrievers
    add_documents_to_retriever(retriever_doc_1, texts_2023, text_summaries_2023, '2023_conocophilips')
    add_documents_to_retriever(retriever_doc_1, tables_2023, table_summaries_2023, '2023_conocophilips')
    add_documents_to_retriever(retriever_doc_1, images_2023, image_summaries_2023, '2023_conocophilips')

    add_documents_to_retriever(retriever_doc_2, texts_2024, text_summaries_2024, '2024_conocophilips')
    add_documents_to_retriever(retriever_doc_2, tables_2024, table_summaries_2024, '2024_conocophilips')
    add_documents_to_retriever(retriever_doc_2, images_2024, image_summaries_2024, '2024_conocophilips_images')

    return retriever_doc_1, retriever_doc_2

# Global retrievers (initialized only once)
retriever_doc_1, retriever_doc_2 = load_data_and_initialize_retrievers()

def process_query(query):
    

    """Retrieve documents based on the user's query."""
    combined_retriever = RunnableParallel(
    retriever1=retriever_doc_1,
    retriever2=retriever_doc_2,
) | RunnableLambda(lambda x: x["retriever1"] + x["retriever2"])  # Merge lists
    chain_with_sources = {
    "context": combined_retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)
    response = chain_with_sources.invoke(query)
    print("Response:", response['response'])
    return response

