import os
import streamlit as st
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

#Field 1: Design Definitions
st.set_page_config(page_title="dev17 Asistanı", page_icon="🤖", layout="centered")

#Field 2: CSS Injection
# 2. BÖLGE: CSS ENJEKSİYONU (Görünüm ayarları)
st.markdown("""
    <style>
    /* 1. Chat Input (Aşağıdaki mesaj yazma kutusu) ayarları */
    div[data-baseweb="base-input"] input::placeholder {
        color: white !important;
        opacity: 0.8;
    }
    div[data-baseweb="base-input"] input {
        color: white !important;
    }
    
    /* 2. Sidebar (Sol Menü) içindeki TÜM yazıları beyaz yapma operasyonu */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

#Field 3: Sidebar and functional buttons
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    uploaded_file = st.file_uploader("Kendi kural dokümanını yükle (.txt)", type=["txt"])
    if uploaded_file:
        with open("dev17_rules.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Dosya başarıyla güncellendi!")
        st.info("Değişikliği görmek için sayfayı yenile (F5)!")
    
    st.divider() #cool_line
    
    #Clear Chat Button
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

    #Advanced Settings(Temperature Slider)
    with st.expander("⚙️ Gelişmiş Ayarlar"):
        temp_val = st.slider("Yaratıcılık (Temperature)", 0.0, 1.0, 0.5)

#Field 4: Main Title
st.title("🤖 dev17 RAG Asistanı")
st.caption("Şirket kuralları veritabanına bağlı resmi yapay zeka asistanı.")

#Field 5: Backend Mechanism (Singleton)
@st.cache_resource(show_spinner="Veritabanı ve LLM Motoru Başlatılıyor...")
def initialize_rag_pipeline(temperature):
    load_dotenv()
    
    #File Reading and Chunking
    loader = TextLoader("dev17_rules.txt", encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
    
    # FAISS Vector Database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    retriever = FAISS.from_documents(chunks, embeddings).as_retriever()
    
    #Brain-Prompt (!Slider'dan gelen sıcaklık değeri buraya bağlanıyor!)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=temperature)
    system_command = (
        "Sen dev17 yazılım şirketinin resmi asistanısın. "
        "Kullanıcının sorularını SADECE sana verilen aşağıdaki şirket kuralları metnine dayanarak cevapla. "
        "Eğer sorunun cevabı kurallarda yazmıyorsa, genel yazılım bilginle cevap ver.\n\n"
        "Şirket Kuralları (Kopya Metni):\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_command),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    #Chain
    rag_pipeline = (
        RunnablePassthrough.assign(context=itemgetter("input") | retriever | format_docs)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    #Memory Management
    session_store = {}
    def get_session_history(session_id: str):
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]
        
    return RunnableWithMessageHistory(
        rag_pipeline, get_session_history, input_messages_key="input", history_messages_key="chat_history"
    )

# Arka planı ayağa kaldır (Slider'dan gelen temperature parametresi ile)
agent = initialize_rag_pipeline(temp_val)
chat_config = {"configurable": {"session_id": "dev17_web_session_01"}}

#Field 6: Frontend State Management and Message Cycle
if "messages" not in st.session_state:
    st.session_state.messages = []

#Previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#user input - answer production
if user_input := st.chat_input("dev17 kuralları hakkında sorunuzu yazın..."):
    
    #show the user message and save to memory
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    #take the answer of assistant
    with st.chat_message("assistant"):
        with st.spinner("dev17 veritabanı taranıyor..."):
            answer = agent.invoke({"input": user_input}, config=chat_config)
            st.markdown(answer)
            
    #save the assistant answer to frontend memory
    st.session_state.messages.append({"role": "assistant", "content": answer})