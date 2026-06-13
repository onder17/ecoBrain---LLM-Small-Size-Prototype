import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load API Key
load_dotenv()

print("1. Reading rules document...")
# Read text file with utf-8 encoding
loader = TextLoader("dev17_rules.txt", encoding="utf-8")
documents = loader.load()

print("2. Splitting text into logical chunks...")
# Divide text into 500-character chunks with 50-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print("3. Building temporary FAISS Vector Database...")
# Initialize embedding model and FAISS store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

print("4. Establishing AI Core and Memory Architecture...")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Define AI personality and instructions
system_command = (
    "Sen dev17 yazılım şirketinin resmi asistanısın. "
    "Kullanıcının sorularını SADECE sana verilen aşağıdaki şirket kuralları metnine dayanarak cevapla. "
    "Eğer sorunun cevabı kurallarda yazmıyorsa, asla kendi mantığınla uydurma, 'Bu konuda bir kural bulunamadı' de.\n\n"
    "Şirket Kuralları (Kopya Metni):\n{context}"
)

# Build prompt with Chat History placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", system_command),
    MessagesPlaceholder(variable_name="chat_history"), # INJECTS MEMORY HERE
    ("human", "{input}"),
])

# Helper function to convert retrieved documents to raw text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL RAG Pipeline with dynamic context assignment
rag_pipeline = (
    RunnablePassthrough.assign(context=itemgetter("input") | retriever | format_docs)
    | prompt
    | llm
    | StrOutputParser()
)

# --- MEMORY MANAGEMENT SYSTEM ---
# Dictionary to store chat histories per user session
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Wrap the pipeline with the Memory Manager
conversational_agent = RunnableWithMessageHistory(
    rag_pipeline,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- INTERACTIVE TESTING STAGE ---
print("\n" + "=" * 60)
print("🤖 dev17 Assistant Initialized! (Type 'q' or 'exit' to quit)")
print("=" * 60 + "\n")

# Provide a unique session ID for this specific chat
chat_config = {"configurable": {"session_id": "dev17_gorkem_session_01"}}

while True:
    # Get user input
    user_input = input("You 👤: ")
    
    # Exit mechanism
    if user_input.lower() in ['q', 'exit', 'quit']:
        print("\n🤖 dev17: Shutting down. Happy coding!")
        break
        
    # Prevent empty API calls
    if not user_input.strip():
        continue

    print("🤖 dev17 is typing...\n")
    
    # Fire the system with user input and session config
    answer = conversational_agent.invoke(
        {"input": user_input},
        config=chat_config
    )
    
    print(f"🤖 dev17 💬: {answer}\n")
    print("-" * 60)