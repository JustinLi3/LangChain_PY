from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import * 
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup   
import requests

URL = "https://ods.od.nih.gov/factsheets/ExerciseAndAthleticPerformance-HealthProfessional/" 
page = requests.get(URL) 
soup = BeautifulSoup(page.content, "html.parser")  
#Check status code 
results = soup.find('body') 
text = results.get_text() 
print(text)

#Check if data has been scraped correctly 
with open('dietary-data.txt','w') as file:
    file.write(text) 

model = ChatOpenAI( 
    api_key="",
    model = "gpt-3.5-turbo-1106", 
    temperature = 0.2
)  


prompt  = ChatPromptTemplate.from_messages([
    ("system", "Provide information about the following with this context: {context}"), 
    ("human", "{input}"), 
])  

parser = StrOutputParser();

#identical to piping, with just the feature to pass in additional documents
chain = create_stuff_documents_chain(
    llm = model,  
    prompt = prompt,
    output_parser= parser 
    doc
)

response = chain.invoke({
    "input" : input("Feel free to ask about any dietary supplement: "),
    "context" : []
}) 

print(response)

#It is very essential that we provide our AI application databases/libraries in order to avoid faulty answers or hallucinations 
#AI can only accurately answer what it is being trained on 



