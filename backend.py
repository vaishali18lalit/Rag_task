import uuid
# from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage

from langchain.retrievers.multi_vector import MultiVectorRetriever
import json
from base64 import b64decode
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser





def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
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
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

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
    with open("chunks_2023.json", "r") as file:
        data = json.load(file)
    texts_2023, tables_2023, images_2023 = data["text_chunks"], data["table_chunks"], data["image_chunks"]

    with open("chunks_2024.json", "r") as file:
        data = json.load(file)
    texts_2024, tables_2024, images_2024 = data["text_chunks"], data["table_chunks"], data["image_chunks"]

    with open("summaries_2023.json", "r") as file:
        data = json.load(file)
    text_summaries_2023, table_summaries_2023, image_summaries_2023 = (
        data["text_summaries"], data["table_summaries"], data["image_summaries"]
    )

    with open("summaries_2024.json", "r") as file:
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
    result_doc_1 = retriever_doc_1.invoke(query)
    result_doc_2 = retriever_doc_2.invoke(query)
    final_result = {"2023": result_doc_1, "2024": result_doc_2}
    # print(type(retriever_doc_1))
#     # with open("final_result.json", "w") as f:
#     #     json.dump(final_result, f, indent=4)


#     # parsed_results = parse_docs(final_result)
#     # # Define the chain for querying and processing the prompt
#     # build_prompt_ser = build_prompt(parsed_results, query)
    chain = (
    {
        "context": retriever_doc_1| RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

    response1 = chain.invoke(query)

    chain_with_sources = {
    "context": retriever_doc_1 | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)

    response = chain_with_sources.invoke(
    query
)

    print("Response:", response['response'])

#     # print("\n\nContext:")
#     # for text in response['context']['texts']:
#     #     print(text.text)
#     #     print("Page number: ", text.metadata.page_number)
#     #     print("\n" + "-"*50 + "\n")
#     # for image in response['context']['images']:
#     #     display_base64_image(image)


    return response

