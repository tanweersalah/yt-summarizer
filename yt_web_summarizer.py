import os
from dotenv import load_dotenv
load_dotenv()
import validators
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain import PromptTemplate

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "DNT": "1",
               "Connection": "close", "Upgrade-Insecure-Requests": "1"}


def get_document(url):
    if validators.url(url):
        if 'youtube' in url or 'youtu.be' in url :
            loader = YoutubeLoader.from_youtube_url(url)
        else : 
            #loader = UnstructuredURLLoader(urls=[url], ssl_verify= False, headers = headers)
            loader = WebBaseLoader(web_path=url)
        return loader.load()
    else :
        

        print('Invalid URL')



def document_splitter(document , chunk_size = 5000, chunk_overlap = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(document)


#groq_api = os.getenv("GROQ_API_KEY")



groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGCHAIN_PROJECT"]
os.environ['LANGCHAIN_TRACING_V2'] = "true"

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

template = """
Please provide the summary of this text : 

speech : {text} , 



"""

prompt = PromptTemplate(input_variable = ["text"], template = template)
final_template = """ 
Provide final summary with these points.
Add a Title, ad bullet points

Speech : {text}
"""

final_prompt = PromptTemplate(input_variables=['text'],template=final_template)


chain = load_summarize_chain(llm=llm , chain_type='map_reduce', map_prompt= prompt , combine_prompt = final_prompt )

# Streamlit app UI
def app_ui():
    st.title("Yotube Video / Webpage Summarizer APP")
    
    
    st.subheader("Enter URL to summarize")
    url = st.text_input("Enter URL")
    
    
    if st.button("Generate Summary"):
        with st.spinner('Generating ...'):
            document = get_document(url)
            splitted_docs = document_splitter(document)
            summary = chain.run(splitted_docs)
            st.subheader("Summary : ")
            st.success(summary)

if __name__ == "__main__":
    app_ui()


