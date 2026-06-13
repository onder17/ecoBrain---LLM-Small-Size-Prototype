# *🚀 ecoBrain | AI Prototyping Lab*

Bu depo, **brainApp** uygulamasının bilişsel altyapısını oluşturacak olan yapay zeka entegrasyonu (ecoBrain) için geliştirilmiş bir Ar-Ge ve prototipleme laboratuvarıdır. 

Proje, Google'ın en güncel **Gemini 3.5 Flash** modellerini ve **LangChain** kütüphanesinin modern LCEL (LangChain Expression Language) mimarisini kullanarak kapalı veri setleri üzerinden bağlama duyarlı (context-aware) asistanlar yaratmayı amaçlar.

---

## 🛠️ Teknoloji Yığını (Tech Stack)
* **LLM:** Google Gemini 3.5 Flash (`gemini-3.5-flash`)
* **Embedding Model:** Google Gemini Embeddings (`gemini-embedding-001`)
* **Orchestration:** LangChain (LCEL Architecture)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Environment:** Python 3.11+, python-dotenv

---

## 📂 Proje Mimarisi ve Modüller

Bu laboratuvar, yapay zeka entegrasyonunun farklı katmanlarını test etmek üzere modüler olarak tasarlanmıştır:

### 1. `llm_test.py` (Prompt Engineering & Core LLM)
Yapay zeka modelinin "karakterini" ve "üslubunu" belirlediğimiz, dinamik `PromptTemplate` yapılarının test edildiği temel modüldür. Modelin API sınırları ve `temperature` gibi parametrelerinin davranışı burada analiz edilmiştir.

### 2. `embedding_test.py` (Vector Mathematics)
NLP (Doğal Dil İşleme) dünyasının kalbi olan Transformer mimarisi test edilmiştir. Metinlerin makine diline (3072 boyutlu sayı matrislerine) nasıl çevrildiği ve kelimeler arası anlamsal mesafelerin (Cosine Similarity) vektör uzayında nasıl konumlandığı bu modülde incelenmektedir.

### 3. `rag_test.py` (The Crown Jewel: RAG & Memory)
**dev17** şirket kurallarını baz alan, tamamen özelleştirilmiş bir yapay zeka asistanı testidir.
* **Retrieval-Augmented Generation (RAG):** İnternet verisi yerine, sisteme yüklenen kapalı bir `.txt` dokümanını okur, `RecursiveCharacterTextSplitter` ile anlamlı parçalara böler ve FAISS vektör veritabanında indeksler.
* **Conversational Memory:** Asistan "balık hafızalı" değildir. `RunnableWithMessageHistory` kullanılarak kullanıcının önceki sorularını aklında tutar ve sohbet bağlamını koparmadan kesintisiz bir deneyim sunar.
* **Zero-Hallucination:** Asistana "Sadece sana verilen metne göre cevap ver" katı kuralı (System Prompt) işlenmiştir.

---
