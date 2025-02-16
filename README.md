# RAG Project

<img width="937" alt="Screenshot 2025-02-16 at 1 12 35 PM" src="https://github.com/user-attachments/assets/f232f581-68b0-44fc-8c5b-1c05bbd070cc" />


## Overview

This project implements Retrieval-Augmented Generation (RAG), combining information retrieval with generative models to enhance response quality. The system retrieves relevant documents from a knowledge base and uses a large language model to generate informed responses.

## Features

Document Retrieval: Uses a vector database for efficient semantic search.

Generative Response: Utilizes an LLM to synthesize relevant answers.

Pipeline Integration: Combines retrieval and generation seamlessly.

Scalability: Designed to handle large datasets efficiently.

## Installation

#### Prerequisites

Python 3.8+

#### Virtual environment (optional but recommended)

#### Setup

#### Clone the repository:

git clone <repository_url>
cd <project_directory>

#### Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

#### Install dependencies:

pip install -r requirements.txt

#### Usage

Start the document retrieval service:

python frontend.py

Run the RAG pipeline:

python main.py --query "Your question here"


## Architecture

Data Ingestion: Load and preprocess documents into a vector database.

Retrieval Module: Searches the vector database for relevant passages.

Generation Module: Uses an LLM to generate a response based on retrieved context.

## Dependencies

LLM Framework: OpenAI / Hugging Face Transformers

Vector Database: ChromaDB

Frameworks: LangChain, FastAPI (if using an API)

API Endpoints (if applicable)

POST /query - Accepts a query and returns a generated response.

## Future Enhancements

Improve retrieval ranking with rerankers

Add fine-tuning for domain-specific knowledge

Implement multi-document summarization

## Results 
### Query and Answer
<img width="943" alt="Screenshot 2025-02-16 at 1 05 16 PM" src="https://github.com/user-attachments/assets/e929fa77-8fb1-40bc-8ed4-b32e19e82be1" />

### Contexual Text
<img width="796" alt="Screenshot 2025-02-16 at 1 05 41 PM" src="https://github.com/user-attachments/assets/d4fa3248-22a3-40d7-a795-1daa213f790d" />

### Contexual Text Metadata
<img width="809" alt="Screenshot 2025-02-16 at 1 05 50 PM" src="https://github.com/user-attachments/assets/a662022b-d8d2-4e4a-ab84-e84ad22d41a7" />

### Contexual Table 
<img width="889" alt="Screenshot 2025-02-16 at 1 06 13 PM" src="https://github.com/user-attachments/assets/a9ea7535-916f-44a9-b91d-521abf62c1e6" />

### Contexual Image
<img width="924" alt="Screenshot 2025-02-16 at 1 07 00 PM" src="https://github.com/user-attachments/assets/fb439fce-10df-40b1-bd51-d78adda6408f" />









