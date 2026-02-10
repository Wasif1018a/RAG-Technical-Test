# Offline RAG Pipeline (Tiny RAG)

A lightweight, fully offline RAG-service using Python, Flask, and local LLM llama3.2:3b (via Ollama).

## Features
- **Offline First**: Entirely CPU-friendly and local; no external API calls.
- **Hybrid Search**: Combines keyword overlap and vector similarity (`sentence-transformers`).
- **Citation Support**: Precise line-level citations (e.g., `[product: L1]`).
- **Prompt-Based Defense**: Security instructions embedded in the system prompt to prevent jailbreaks.
- **llama3.2:3b**: Used for its strong reasoning and comprehension abilities within limited memory while running locally.

## Design Choices
- **Hybrid Retrieval**: Chosen to compensate for small vector model limitations by ensuring exact keyword matches are prioritized. 
- **In-Memory Store**: Since the document set is small, a full vector database was swapped for a simple in-memory numpy-based index for maximum speed and simplicity.
- **Strict Prompting**: The model is instructed to strictly adhere to the context and refuse hallucinations or instructions to "ignore" rules.

## Hybrid Search Mechanism
The system uses a weighted combination of two search methods to find the most relevant documents:
1.  **Keyword Search**: Performs a basic word overlap count between the query and documents.
2.  **Vector Search**: Uses `all-MiniLM-L6-v2` embeddings and cosine similarity to understand semantic meaning.
3.  **Normalization**: Both scores are normalized using Min-Max scaling to a range of [0.0, 1.0].
4.  **Combined Score**: The final score is the sum of these normalized values (Max theoretical score is 2.0).
5.  **Threshold**: Chunks are only included in the LLM context if their combined score exceeds the **current threshold of 0.1** which can be changed/optimized.

## Setup & How to Run

1. **Prerequisites**:
   - Install [Ollama](https://ollama.com/) and download `llama3.2:3b`.
   - Install Python 3.9+.

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Local Embedding Model Setup**:
   Run the following script to download and cache the embedding model locally:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   model.save("D:/RAG_Technical_Test/local_models/all-MiniLM-L6-v2")
   ```
   Ensuring the `EMBEDDING_MODEL_NAME` path in `config.py` matches your local storage location.

4. **Running the Server**:
   ```bash
   # Starts server on http://localhost:8000
   python server.py
   ```

5. **Testing the API**:

   ### Option 1: cURL
   ```bash
   curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "", "top_k": 3}'
   ```

   ### Option 2: Postman
   1. Set request type to **POST**.
   2. Enter URL: `http://localhost:8000/ask`.
   3. In the **Headers** tab, add `Content-Type: application/json`.
   4. In the **Body** tab, select **raw** and **JSON**, then enter:
      ```json
      {
          "question": "",
          "top_k": 3
      }
      ```

6. **Verification**:
   ```bash
   # Runs a test suite against the live server
   python verify.py
   ```

## Future Improvements
- **VectorDB Setup**: For large documents storage.
- **Conversation Buffer Memory**: Integrating a memory module to allow the model to remember previous turns in a multi-turn chat.
- **LangChain/LangGraph**: Transitioning to an orchestration framework for multi-agent support and low-code scalability.

## AI Usage Disclosure
This repository was developed with Google's AntiGravity AI assistance used for:
- **Repo Structuring**: Scaffolding the codebase with appropriate files and folders.
- **Boilerplate Implementation**: Creating dummy classes and functions as a starting point for further implementation.
- *Debugging and Code Quality*: Used as a support tool for debugging and improving code quality.