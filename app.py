import streamlit as st
from langsmith import Client

client = Client()
import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
# os.environ['LANGCHAIN_TRACING_V2']=true
# os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
# os.environ['LANGCHAIN_API_KEY']="ls__d676afea5dc04fa1837a8371e9a4d4de"
# os.environ['LANGCHAIN_PROJECT']="Python-code-generator"
template = """generate synthetically correct python code for the given input question

question: {input}

Instruction:
1.Review the question carefully.
2.Generate synthetically correct python code for the question.
3.Return only the python code one for the input question without any extras as output.
4.Also debug if any code with error is given as input

Python Code: {output}"""

prompt = PromptTemplate(template=template, input_variables=["input","output"])
repo_id = "bigcode/starcoder"  
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1,"max_new_tokens":1000, "repetition_penalty": 1.13, "stop_sequences": ["def "],"num_return_sequences": 2}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)


st.set_page_config(page_title="Code Llama demo", page_icon="🤖")
st.title("Python Code Generation-test")

input = st.text_area("Type your Question")

if st.button("Generate Code") and input:
        with st.spinner("Generating Code....."):
            response = llm_chain.run({"input":input,"output":""})
            st.write(response)

