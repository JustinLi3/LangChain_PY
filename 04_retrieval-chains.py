from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
#To combine multiple documents into a single coherent chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Retrieve Data
def get_docs(): 
    #Create instance of WebBaseLoader and initializes it with url  
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/') 
    #Retrieve content into docs
    docs = loader.load()
    #Initialize a text splitter with settings chunks of 200 characters with 20 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    #return a list of chunks
    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs): 
    #Create an instance of embeddings 
    embedding = OpenAIEmbeddings() 
    #Organize documents into vectorStore with pre-computed data: embedding 
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

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    
    #transforms vector store into a search tool 
    retriever = vectorStore.as_retriever()
    #Creates chain with the search tool 
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    #
    return retrieval_chain


docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})

print(response)



