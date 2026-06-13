import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

word1 = "Galatasaray"
word2 = "Okan Buruk"
word3 = "Spring Boot Microservices"

print("The words are transforming to transformer matrixes...\n")

#Words to vectors processes
vector1 = embeddings.embed_query(word1)
vector2 = embeddings.embed_query(word2)
vector3 = embeddings.embed_query(word3)

print("-" * 60)

#Examine the final results
print(f"'{word1}' kelimesi yapay zeka için {len(vector1)} boyutlu bir sayı dizisidir.")
print(f"İşte bu dizinin ilk 5 rakamı:")
print(vector1[:5])
print("\n")

print("-" * 60)

print(f"'{word2}' kelimesi yapay zeka için {len(vector2)} boyutlu bir sayı dizisidir.")
print(f"İşte bu dizinin ilk 5 rakamı:")
print(vector2[:5])
print("\n")

print("-" * 60)
print(f"'{word3}' kelimesi yapay zeka için {len(vector3)} boyutlu bir sayı dizisidir.")
print(f"İşte bu dizinin ilk 5 rakamı:")
print(vector3[:5])
print("\n")
print("-" * 60)