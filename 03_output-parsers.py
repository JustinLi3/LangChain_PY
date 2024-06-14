from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate   
#Import output parser
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser 
from langchain_core.pydantic_v1 import BaseModel, Field

model = ChatOpenAI( api_key="",  model = "gpt-3.5-turbo-1106", temperature = 0.7, verbose=False)
 
def json_output_parser(): 
    prompt = ChatPromptTemplate.from_messages([                                               #tell ai to use these formatting instructions
        ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"), 
        ("human", "{phrase}")
    ])  
    #Set up expected behavior with Base Model and Fields to provide what the AI should expect and decipher
    class Person(BaseModel): 
        name: str = Field(description="the name of the person") 
        age: int = Field(description="the age of the person")  
        email: str = Field(description="the email of the person") 
        car: list = Field(description="list of attributes that describe his car according to the model")
    
    #put in the basemodel to be expected
    parser = JsonOutputParser(pydantic_object=Person) 
    chain = prompt | model | parser 
    return chain.invoke({
        "phrase" : "Justin is 40 years old and his email is lijustin83 gmail. He drives a 2017 black camry se", 
        #dynamically generate instructions to hook into a json (dictionary)
        "format_instructions" : parser.get_format_instructions() 
    })
 

def list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate list of 10 attributes to the following. Please seperate with commas."), 
        ("human", "{input}")
    ])  
    # parser = StrOutputParser()  
    parser = CommaSeparatedListOutputParser()
    #pass result through parser, essentially parsers help define a type of the output 
    chain = prompt | model | parser
    return chain.invoke({"input": input("Please enter something you would like to know: ")}) 
 
print(json_output_parser()) 