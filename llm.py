from dotenv import load_dotenv
from langchain_openai import ChatOpenAI   
load_dotenv() #load all env variables

llm = ChatOpenAI(
    model = "gpt-3.5-turbo", #specify model 
    temperature = 0.2,   #temp 0 is strict and factual and 1 being creative 
    max_tokens = 1000,   #750 words 
    verbose = True  #debug output from model 
    ) #automatically look for OPENAI_API_KEY

#invoke to send in prompt and recieve a response 
response = llm.invoke("Hello, how are you?")  
#batch to send in multiple prompts and recieve a list of responses
# responses = llm.batch(["Hello how are you?", "What is the weather like today"])  
#stream to return a list of chunks  
# test = llm.stream("How does the implementation of Content Delivery Networks (CDNs) like Varnish and NGINX compare in terms of cost, performance, and feature set? Specifically, what are the cost differences between their free and paid versions, and how do these differences impact small to medium-sized enterprises? Additionally, what performance metrics are essential for evaluating the effectiveness of these CDNs, and how do Varnish and NGINX perform against these metrics in real-world scenarios? Moreover, what are the benefits and potential challenges of combining Varnish and NGINX in a CDN strategy, and how can these tools be integrated to optimize web performance and reliability? Finally, what are the advantages of utilizing Geo DNS in conjunction with these CDNs, and how does this combination influence the overall user experience and global reach of a website?")
# for chunk in test:
#     print(chunk.content, end = "", flush=True)