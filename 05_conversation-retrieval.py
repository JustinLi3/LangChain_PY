from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage

def get_docs(): 
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/') 
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs): 
    embedding = OpenAIEmbeddings() 
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    #top 3 vector chunks that closely relate to input
    retriever = vectorStore.as_retriever(search_kwargs = {"k":3}) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def processQuestion(chain, question): 
    response = chain.invoke({
    "input": question,
    })
    #response would, because of retrieval chain, return a dictionary of input, context, and answer
    return response["answer"]

#Similar to public static void main..
if __name__ == '__main__': 
    docs = get_docs()
    vectorStore = create_vector_store(docs)
    chain = create_chain(vectorStore)    
    chatHistory = [] 
    
    print("Assistant: Hello, how are you? Please feel free to ask me anything and type 'exit' to leave.")
    while True: 
        userInput = input("You: ")  
        if(userInput.lower() =="exit"): 
            break  
        chatHistory.append(userInput)
        print("Assistant: ", processQuestion(chain,userInput))





