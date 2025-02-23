##integarting the code with  openAI API
import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
gemini=os.getenv("gemini")



import streamlit as st



#streamlit framework

st.title("LangChain with OpenAi API")
input_text=st.text_input("Search the topic u want")

#openAI LLMS
llm=GoogleGenerativeAI(model="gemini-pro",google_api_key=gemini)


if input_text:
    st.write(llm(input_text))
