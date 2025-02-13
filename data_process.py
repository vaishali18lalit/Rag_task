import os
import json
from unstructured.partition.pdf import partition_pdf
from config import Config  # Import Config class to use environment variables
from huggingface_hub import login
from PIL import Image as PILImage
import io
import base64
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
import json 




# Login to Hugging Face
def login_huggingface():
    huggingface_token = Config.HUGGINGFACE_TOKEN
    login(token=huggingface_token)

# Get API keys from environment (Config class)
def get_api_keys():
    return {
        "openai": Config.OPENAI_API_KEY,
        "groq": Config.GROQ_API_KEY,
        "langchain": Config.LANGCHAIN_API_KEY,
        "langchain_tracing": Config.LANGCHAIN_TRACING_V2
    }

def summarize_text_chunk(element):
    """
    Function to summarize a given text or table chunk using the ChatGroq model.
    Args:
    element (str): The text or table chunk to be summarized.
    Returns:
    str: The summarized text.
    """
    # Prompt template for summarizing
    prompt_text = """
    You are an assistant tasked with summarizing text chunks while preserving key details and important quantitative metrics.
    Provide a detailed yet concise summary that captures essential information, key points, and context.
    Ensure completeness while avoiding unnecessary length.
    Respond only with the summary, without any additional comments.
    Table or text chunk: {element}
    """
    # Create a prompt from the template
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # Set up the model for summarization
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    # Define the summarization chain
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Return the result of the summarization
    summaries = summarize_chain.batch(element, {"max_concurrency": 2})

    return summaries

def generate_image_summaries(images, prompt_template=None):
    """
    Function to generate detailed image summaries for a list of images.
    Args:
    images (list): A list of image data (base64 encoded).
    prompt_template (str): Optional. Custom prompt template for image description.
    Returns:
    list: List of summaries for each image.
    """
    # Default prompt template if not provided
    if prompt_template is None:
        prompt_template = """Describe the image in detail. For context,
                              the image is part of performance reports of a company called Conoco Phillips. Be specific about graphs, such as bar plots. If you feel that it is an image of a person or signature of a person, simply mention it as a person's image or person's signature."""
    # Prepare the message structure for the prompt
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    # Initialize the prompt template with the message
    prompt = ChatPromptTemplate.from_messages(messages)
    # Set up the model chain
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    # Generate summaries for all images in the list
    image_summaries = chain.batch(images)
    return image_summaries

# Partition a PDF and extract chunks
def partition_pdf_file(file_path, max_characters=5000, chunking_strategy="by_title"):
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
        combine_text_under_n_chars=1300,
        new_after_n_chars=3000
    )

# Get images and tables from chunks
def extract_images_and_tables(chunks):
    images_b64 = []
    tables = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
                if "Table" in str(type(el)):
                    tables.append(el.metadata.text_as_html)
    return images_b64, tables

def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    image = PILImage.open(io.BytesIO(image_data))
    image.show()  # This will display the image using the default image viewer

# Process PDF files
def process_pdfs(file_path):
    all_texts = []
    all_images = []
    all_tables = []
    print(f"Processing {file_path}...")
    # Partition the PDF into chunks
    chunks = partition_pdf_file(file_path)
    print(f"Number of chunks: {len(chunks)}")
    # Extract images and tables
    images, tables = extract_images_and_tables(chunks)
    # Collect all data
    for chunk in chunks:
        all_texts.append(chunk)

    all_images.extend(images)
    all_tables.extend(tables)
    # print(all)
    return all_texts, all_images, all_tables

# Main processing
def main():
    # Login to Hugging Face
    login_huggingface()

    # Get the API keys from environment variables
    api_keys = get_api_keys()

    # Define the PDF file paths
    file_paths = [
        "/Users/vaishalilalit/Desktop/RAG_task/data/2023-conocophillips-aim-presentation_compressed (1).pdf",
        "/Users/vaishalilalit/Desktop/RAG_task/data/2024-conocophillips-proxy-statement.pdf"
    ]
    
    # # Process the PDFs
    # # texts_2023, images_2023, tables_2023 = process_pdfs(file_paths[0])
    texts_2024, images_2024, tables_2024 = process_pdfs(file_paths[1])
    chunks_2024 = {
        "text_chunks": [text.to_dict() for text in  texts_2024],
        "table_chunks": tables_2024,
        "image_chunks": images_2024,
    }

    with open('chunks_2024.json', 'w') as f_chunks_2024:
        json.dump(chunks_2024, f_chunks_2024, indent=4)


    # # Summarize text
    # text_summaries_2023 = summarize_text_chunk(texts_2023)
    # # Summarize tables
    # # tables_html = [table.metadata.text_as_html for table in tables]
    # table_summaries_2023 = summarize_text_chunk(tables_2023)
    # with open("chunks_2024.json", "r") as file:
    #     data = json.load(file)  # Loads JSON as a Python dictionary
    
    # texts_2024 = data["text_chunks"]
    # tables_2024 = data["table_chunks"]
    # images_2024 = data["image_chunks"]

    # Summarize text
    text_summaries_2024 = summarize_text_chunk(texts_2024)
    # # Summarize tables
    table_summaries_2024 = summarize_text_chunk(tables_2024)
    # # Generate image summaries
    # image_summaries_2023=generate_image_summaries(images_2023)
    image_summaries_2024=generate_image_summaries(images_2024)

    # chunks_2023 = {
    #     "text_chunks": [text.to_dict() for text in  texts_2023],  # Convert objects to strings
    #     "table_chunks": tables_2023,
    #     "image_chunks": images_2023,  # Images are already in base64, so should be fine
    # }



    # # Define summaries for 2023 and 2024 in a dictionary format
    # summaries_2023 = {
    #     "text_summaries": text_summaries_2023,
    #     "table_summaries": table_summaries_2023,
    #     "image_summaries": image_summaries_2023,
    # }

    summaries_2024 = {
        "text_summaries": text_summaries_2024,
        "table_summaries": table_summaries_2024,
        "image_summaries": image_summaries_2024,
    }

    # # Save chunks_2023 to a JSON file
    # with open('chunks_2023.json', 'w') as f_chunks_2023:
    #     json.dump(chunks_2023, f_chunks_2023, indent=4)

    # Save chunks_2024 to a JSON file
    

    # # Save summaries_2023 to a JSON file
    # with open('summaries_2023.json', 'w') as f_summaries_2023:
    #     json.dump(summaries_2023, f_summaries_2023, indent=4)

    # Save summaries_2024 to a JSON file
    with open('summaries_2024.json', 'w') as f_summaries_2024:
        json.dump(summaries_2024, f_summaries_2024, indent=4)


# Directly call the main function
main()
