from dotenv import load_dotenv
from langchain_openai import ChatOpenAI     
#Import prompt
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

#Instantiate Model
llm = ChatOpenAI(  
    api_key= "",
    model = "gpt-3.5-turbo-1106", 
    temperature = 0.7,  
    )  

#Prompt Template 
#prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")  

#Prompt From messages (list of messages, system to prime the model (what you want it to do))
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms to the following word. Return them seperated by commas"),
        ("human", "{Similar}")
    ]
)
 
 
 #Create LLM Chain (allow us to combine objects together, take out prompt and pipe into llm )
chain = prompt | llm 
 
response = chain.invoke({"Similar" : "Chai"}) #invoke chain but pass in value for subject through a dictionary 
print(response)
