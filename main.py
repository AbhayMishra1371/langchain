import os
from constants import gemini_key
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Set API Key
os.environ["GOOGLE_API_KEY"] = gemini_key

# Streamlit framework
st.title('Langchain Chatbot')
input_text = st.text_input('Search here')

# Initialize the LLM (fixing the order issue)


# Prompt Template (fixing typo)
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about celebrity {name}'
)
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Define the LLMChain
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

# Process user input
if input_text:
    st.write(chain.run({'name': input_text}))
