import torch
from transformers import pipeline

# This works on a base Colab instance.
# Pick a larger checkpoint if you have time to wait and enough disk space!
checkpoint = "facebook/opt-6.7b"
# generator = pipeline("text-generation", model=checkpoint, device_map="auto", torch_dtype=torch.float16)
# generator = pipeline("text-generation", model=checkpoint, torch_dtype=torch.float16)
generator = pipeline("text-generation", model=checkpoint)

# Perform inference
res = generator("Can you write Cypress tests? Provide an example.")

print(res)



#%%
