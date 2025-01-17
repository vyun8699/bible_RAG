#solve chromedb
'''__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3'''

#environment
from dotenv import load_dotenv
load_dotenv()

#imports for chroma and embeddings
import chromadb
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

#Chats
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain

# Streamlit imports
import streamlit as st

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Page config
st.set_page_config(page_title="Bible QA", layout="wide")

# Add clear history button in top right (smaller)
col1, col2, col3 = st.columns([8, 1, 1])
with col3:
    if st.button("Clear History", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Create tabs
tab1, tab2 = st.tabs(["Chat", "History"])

# get embeddings from chroma
embeddings_model = OpenAIEmbeddings()

client = chromadb.PersistentClient(path='assets/bible/chromadb')
vectorstore = Chroma(client=client, embedding_function=embeddings_model, persist_directory='assets/bible/chromadb')

#define llm
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

#define retriever
document_content_description = "The holy bible"

retriever = vectorstore.as_retriever(
    search_type = 'mmr', 
    search_kwargs={'k':5, 'fetch_k':20}
    )

# Define QA prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, do not try to make up an answer. "
    "Use five sentences maximum. "
    "Keep the answer as concise as possible. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Define history aware retrieval 
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Combine both elements and define rag chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat tab content
with tab1:
    st.title("Bible Q&A")
    
    # Question input
    question = st.text_input("Ask a question about the Bible:", key="question_input")
    
    if st.button("Ask"):
        if question:
            # Get the response
            with st.spinner("Thinking..."):
                result = rag_chain.invoke({
                    "input": question,
                    "chat_history": [
                        msg for msg in st.session_state.chat_history
                    ]
                })
                
                # Display the answer
                st.write("### Answer:")
                st.write(result["answer"])
                
                # Display the context with metadata
                st.write("### Context:")
                for doc in result["context"]:
                    metadata = doc.metadata
                    reference = f"{metadata['book']} {metadata['chapter']}:{metadata['verse']}"
                    st.markdown(f"• **{reference}** - {doc.page_content}")
                
                # Update chat history
                st.session_state.chat_history.extend([
                    HumanMessage(content=question),
                    AIMessage(content=result["answer"])
                ])

# History tab content
with tab2:
    st.title("Chat History")
    
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.write("🙋‍♂️ **You:** " + msg.content)
            else:
                st.write("🤖 **Assistant:** " + msg.content)
            st.write("---")
    else:
        st.write("No chat history yet. Start asking questions in the Chat tab!")

if __name__ == "__main__":
    pass
