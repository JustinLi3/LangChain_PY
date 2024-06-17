from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain  
from langchain_core.messages import HumanMessage, AIMessage  
from langchain.chains.history_aware_retriever import create_history_aware_retriever

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),  
        #This is how you pass in a placeholder for background knowledge 
        #Placeholder/Temp storage for messages, points to the list to use all the notes inside when responding 
        MessagesPlaceholder(variable_name="chatHistory"),
        ("human","{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    #fetch top 3 vector chunks that closely relate to input, HOWEVER, we want to attach chat history too  
    retriever = vectorStore.as_retriever(search_kwargs = {"k":3})  
    retrieverPrompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name= "chatHistory"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to get information relevant to the conversation")
    ])
    historyAwareRetriever = create_history_aware_retriever( 
        llm = model,  
        retriever = retriever, 
        prompt = retrieverPrompt,
    )
    retrieval_chain = create_retrieval_chain(historyAwareRetriever, document_chain)
    return retrieval_chain

def processQuestion(chain, question, chatHistory): 
    response = chain.invoke({
    "input": question,  
    #Make sure that when processing the question that you include this for history
    "chatHistory" : chatHistory
    })
    #response would, because of retrieval chain, return a dictionary of input, context, and answer
    return response["answer"]

#Similar to public static void main..
if __name__ == '__main__': 
    docs = get_docs()
    vectorStore = create_vector_store(docs)
    chain = create_chain(vectorStore)    
    chatHistory = [  
                   
        #this is the roleplaying thing, kind of gives a background and placeholder
        # HumanMessage(content = 'Hello'),
        # AIMessage(content='Hello, how can I assist you?'), 
        # HumanMessage(content = 'My name is Justin')
    ] 
    
    print("Assistant: Hello, how are you? Please feel free to ask me anything and type 'exit' to leave.")
    while True: 
        userInput = input("You: ")  
        if(userInput.lower() =="exit"): 
            break   
        response = processQuestion(chain,userInput, chatHistory) 
        chatHistory.append(HumanMessage(content = userInput))
        chatHistory.append(AIMessage(content= response)) 
        print("Assistant:", response)





