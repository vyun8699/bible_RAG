# Biblechats (Bible Based RAG Model)

A chatbot powered by OpenAI that uses the King James Version (KJV) of the Holy Bible as its source of truth through retrieval augmented generation.

## Data Source

The system uses the King James Version (KJV) of the Holy Bible as its primary data source. The biblical text is naturally structured in a hierarchical format:

- Books (e.g., Genesis, Exodus, etc.)
- Chapters within each book
- Verses within each chapter

This natural segmentation provides ideal chunks for text embedding and retrieval, maintaining the contextual integrity of the scripture while allowing for precise information retrieval.

## Technical Implementation

### Embeddings Generation

The system uses LangChain's embedding API to convert biblical text into vector representations. These embeddings capture the semantic meaning of each verse chunk, enabling:

- Semantic search capabilities
- Context-aware retrieval
- Meaningful similarity comparisons

The embeddings are persistently stored in a ChromaDB database (`assets/bible/chromadb/`), allowing for:
- Fast retrieval
- Efficient similarity search
- Persistent storage between sessions

### Chatbot Architecture

The chatbot leverages the OpenAI API with the following key features:

1. **Source of Truth**: Uses the KJV Bible embeddings as its knowledge base
2. **Temperature Setting**: Set to 0 for consistent, factual responses
3. **Context Awareness**: Maintains chat history to provide contextually relevant responses
4. **RAG Implementation**: 
   - Uses Maximum Marginal Relevance (MMR) search strategy to retrieve diverse, relevant Bible verses
   - Implements a history-aware retrieval system that reformulates questions based on chat context
   - Retrieves top 5 most relevant verses from a candidate pool of 20
   - Displays source verses with precise book, chapter, and verse references
   - Uses a specialized prompt template to ensure concise, accurate responses

## Setup and Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/[username]/bible_RAG.git
cd bible_RAG
\`\`\`

2. Install required dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

The application runs on Streamlit. To start the chatbot:

\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

This will launch a web interface where you can interact with the Bible-aware chatbot.



