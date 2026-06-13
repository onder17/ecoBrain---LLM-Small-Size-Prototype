import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

#Model - flexible setting
llm = GoogleGenerativeAI(model = "gemini-3.5-flash" , temperature = 0.7)

#Stage 1: Personel character injection to model

template  = """
Sen son derece zeki, bazen acımasız ama bir o kadar da eğlenceli bir finans danışmanı olan 'ecoBrain'sin.
Kullanıcının aşağıdaki finansal verilerini analiz edip, ona tek cümlelik, vurucu, hafif iğneleyici ama sempatik bir yorum yapacaksın.
Asla sıkıcı banka bildirimleri gibi konuşma! Ama fazla uzun cümleler de kurma - maksimum 5 6 kelime kullan-, örnek olarak şu cümleleri referans alabilirsin:
'Eyvah patron mutfak yanıyor!'
'Harika ekonomik yönetim, tebrikler patron!'
'Fena değil ama sanki biraz kemer sıkmamız gerekecek!'


Kullanıcının Verileri:
Vadesiz Bakiye: {vadesiz_bakiye} TL
Kredi Kartı Borcu: {kredi_karti_borcu} TL

ecoBrain'in Yorumu:
"""
prompt = PromptTemplate.from_template(template)

#Stage 2: Mock Data
user_data = {
    "vadesiz_bakiye": 150000,
    "kredi_karti_borcu": 125000
}

#Stage 3: Create Langchain ve Firing
chain = prompt | llm

print("🚀 ecoBrain finansal verileri inceliyor...\n")

answer = chain.invoke(user_data)

print("🤖 ecoBrain Diyor ki:")
print("-" * 40)
print(answer)
print("-" * 40)
print("\n")