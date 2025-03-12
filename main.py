import os
from constants import gemini_key
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

# Set API Key
os.environ["GOOGLE_API_KEY"] = gemini_key

# Streamlit framework
st.title('Langchain Chatbot')
input_text = st.text_input('Search here')

# Initialize the LLM (fixing the order issue)




#Memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='description_history')

# Prompt Template (fixing typo)
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about celebrity {name}'
)
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Define the LLMChain
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True,output_key='person',memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template='when was {person} born'
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True,output_key='dob',memory= dob_memory)

Third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template='mention 5 major events happend around {dob} in the world'
)
chain3 = LLMChain(llm=llm, prompt=Third_input_prompt, verbose=True,output_key='description',memory=descr_memory)


parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

# Process user input
if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memory.buffer)