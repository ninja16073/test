from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_KEY"]



def construct_index(directory_path):
    # set maximum input size
    max_input_size = 100
    # set number of output tokens
    num_outputs = 100
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-babbage-001", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

construct_index("context_data/data")

import gradio as gr
def ask_ai(AskAnythingAboutShreyaAndIshan):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True: 
        response = index.query(AskAnythingAboutShreyaAndIshan)
       
        return response
      
demo = gr.Interface(fn=ask_ai, inputs="text", outputs="text")
demo.launch()
