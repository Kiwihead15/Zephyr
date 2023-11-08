
import os
import chainlit as cl
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2)
}

local_llm = CTransformers(model='C:\\Users\\pbiosca\\AI\\Projects\\models\\zephyr-7b-beta.Q3_K_S.gguf', config=config)  # Path to the local Zephyr-7B model file
#local_llm = CTransformers(model='TheBloke/zephyr-7B-beta-GGUF', model_file='zephyr-7b-beta.Q2_K.gguf', config=config)

template = """Question: {question}

Answer: Please refer to factual information and don't make up fictional data/information.
"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['question'])
    llm_chain = LLMChain(prompt=prompt, llm=local_llm, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()

# %%
#! chainlit run question-answer_Zepphyr_7B.py


