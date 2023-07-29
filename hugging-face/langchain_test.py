import sys
sys.path.append('<path_to_langchain_installation>')

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from llama_cpp import Llama
import time

start = time.time()

llm = Llama(model_path="./ggml-model-q4_0.bin")
output = llm("Q: write Cypress tests example. A: ", max_tokens=256, stop=["Q:", "\n"], echo=False)
print(output["choices"][0]["text"])
print((time.time() - start)/60)


#%%
