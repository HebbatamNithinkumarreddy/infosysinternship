Competitor Monitoring & Pricing Optimization using LLMs

AI-powered competitor intelligence for e-commerce teams using Retrieval-Augmented Generation (RAG), Google Generative AI embeddings, and Groq LLaMA 3.3.

 Overview

This project leverages Learning Technologies & Large Language Models (LLMs) to:

Monitor competitor pricing & discount strategies

Perform sentiment analysis of customer feedback

Detect real-time market trends

Deliver actionable, data-driven insights to optimize pricing & promotions

 Features

Document Processing – Supports .txt, .md, .pdf

RAG-based QA – Combines FAISS vector search + Groq LLaMA 3.3

Google Generative AI Embeddings – High-quality semantic search

Configurable Pipeline – Easy to adapt to different datasets and queries

Tech Stack

Python 3.10+

LangChain

Google Generative AI

Groq (LLaMA 3.3)

FAISS

dotenv

 Project Structure
competitor-tracker/
│── my_docs/            # Your input documents (.txt, .md, .pdf)
│── faiss_index/        # Saved FAISS index
│── rg.py               # Main Retrieval-Augmented Generation pipeline
│── groq_test.py        # Simple Groq API test
│── requirements.txt    # Python dependencies
│── myenv/.env          # API keys (excluded from git)
│── README.md           # Documentation

 Setup & Installation
1. Clone the Repository
git clone https://github.com/yourusername/competitor-tracker.git
cd competitor-tracker

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Configure API Keys

Create a .env file inside myenv/:

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key

 Usage
1. Add Your Documents

Place your .txt, .md, .pdf files inside my_docs/.

2. Run Retrieval-Augmented Generation
python rg.py


Scans and indexes documents

Uses FAISS for semantic retrieval

Answers the question defined in rg.py (QUESTION variable)

3. Test Groq LLM Directly
python groq_test.py


Sends a sample prompt ("i am good at using rag") to Groq LLaMA 3.3

Prints the response

Requirements
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-google-genai>=0.2.0
langchain-groq>=0.1.0
faiss-cpu>=1.7.4
pypdf>=3.9.0
python-dotenv>=1.0.0
groq>=0.3.0
 Example Output
Question: what are building blocks of mobile application development.

Answer:
The building blocks include UI/UX design, APIs, backend services, data storage, security, and deployment pipelines.

Sources:
mobile_dev_notes.pdf (page 3)

Future Enhancements

Streamlit Dashboard for visualization

Live API integration (Amazon, Flipkart, etc.)

Automated price optimization engine
License

This project is licensed under the MIT License.
