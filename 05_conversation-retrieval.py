import os 
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
from typing import List, Union, Generator, Iterator
import subprocess 



class Pipeline: 
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "Python Code Pipeline" 
        self.chatHistory = [  
        #this is the roleplaying thing, kind of gives a background and placeholder
        # HumanMessage(content = 'Hello'),
        # AIMessage(content='Hello, how can I assist you?'), 
        # HumanMessage(content = 'My name is Justin')
        ]   
        self.docs = None 
        self.chain = None 
        self.vectorStore = None
        pass

    async def on_startup(self):
        # This function is called when the server is started. 
        self.docs = get_docs()
        self.vectorStore = create_vector_store(self.docs)
        self.chain = create_chain(self.vectorStore)    
    
        print("Assistant: Hello, how are you? Please feel free to ask me anything and type 'exit' to leave.")
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def execute_python_code(self, code):
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            stdout = result.stdout.strip()
            return stdout, result.returncode
        except subprocess.CalledProcessError as e:
            return e.output.strip(), e.returncode

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)  
        response = processQuestion(self.chain,user_message, self.chatHistory) 
        self.chatHistory.append(HumanMessage(content = user_message))
        self.chatHistory.append(AIMessage(content= response)) 
        print("Assistant:", response)

        if body.get("title", False):
            print("Title Generation")
            return "Python Code Pipeline"
        else:
            stdout, return_code = self.execute_python_code(user_message)
            return stdout 
        
        
def get_docs(): 
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/') 
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = text_splitter.split_documents(docs) 
    print("DOCS RAN")

    return splitDocs

def create_vector_store(docs): 
    embedding = OpenAIEmbeddings(
        api_key = os.getenv("OPEN_API_KEY")
        ) 
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    print("VS RAN")
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        api_key = os.getenv("OPEN_API_KEY"),
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
    print("RETRIEVAL CHAIN RAN")

    return retrieval_chain

def processQuestion(chain, question, chatHistory): 
    response = chain.invoke({
    "input": question,  
    #Make sure that when processing the question that you include this for history
    "chatHistory" : chatHistory
    })
    #response would, because of retrieval chain, return a dictionary of input, context, and answer
    print("QUESTION")

    return response["answer"]

#Similar to public static void main..

    





