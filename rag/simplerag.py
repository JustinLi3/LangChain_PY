#Data source loading

#Text based loader
from langchain_community.document_loaders import TextLoader 
loader1 = TextLoader("rag/rag_pipeline.txt") 
text_document = loader1.load() 
# print(text_document) 

import os
os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_API_KEY") 


#Text based loader
from langchain_community.document_loaders import WebBaseLoader  
import bs4 
loader2 = WebBaseLoader("https://www.californiastrawberries.com/can-strawberries-improve-heart-health/") 
web_document = loader2.load()
# print(web_document)
 
#PDF loader
from langchain_community.document_loaders import PyPDFLoader 
loader3 = PyPDFLoader("rag/formadv.pdf")   
pdf_document = loader3.load() 
# print(pdf_document)

from langchain.text_splitter import RecursiveCharacterTextSplitter 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 30) 
chunks = text_splitter.split_documents(pdf_document) 

#Vector store and embeddings
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma  
#get first 20 chunks, using the embeddings to vectorize and store into vector store
db = Chroma.from_documents(chunks[:20],OpenAIEmbeddings()) 

query = input("Please enter a question: ")
result = db.similarity_search(query) 
print(result[0].page_content)