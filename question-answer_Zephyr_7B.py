
import os
import chainlit as cl
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 1,
    "top_k": 5,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2)
}
#loading the model from the local directory
local_llm = CTransformers(model='C:\\Users\\pbiosca\\AI\\Projects\\models\\zephyr-7b-beta.Q6_K.gguf', config=config)  # Path to the local Zephyr-7B model file

#loding the model from the Huggingface-hub
#local_llm = CTransformers(model='TheBloke/zephyr-7B-beta-GGUF', model_file='zephyr-7b-beta.Q2_K.gguf', config=config)

template = """Question: {question}

Answer: Please refer to factual information and don't make up fictional data/information.
"""

@cl.on_chat_start
def main():
    local_llm = CTransformers(model='C:\\Users\\pbiosca\\AI\\Projects\\models\\zephyr-7b-beta.Q6_K.gguf', config=config)
    
    prompt = PromptTemplate(template=template, input_variables=['question'])
    
    runnable = prompt | local_llm | StrOutputParser()
    
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message: cl.Message):
    # sets up an instance of Runnable with a custom ChatPromptTemplate for each chat session. 
    # The Runnable is invoked everytime a user sends a message to generate the response.
    runnable = cl.user_session.get("runnable") # type: Runnable
    
    # Prepare the message for streaming
    msg = cl.Message(content="")
    
    # The callback handler is responsible 
    # for listening to the chain’s intermediate steps and sending them to the UI
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    # Send a response back to the user and close the message stream
    await msg.send()

# %%
#! chainlit run question-answer_Zephyr_7B.py -w


