# Local-Language-Model-Project
In this project 2 local language models, openai-community/gpt2 and openai-community/gpt2-medium, are loaded from HuggingFace.

GPT 2 has 137 million parameters, while GPT 2 Medium has 380 million parameters.

I chose these two local language models because they are so small and my computer had too much trouble running the larger models on HuggingFace.

## Steps
Have python 3.8 or higher installed (I am personally using 3.12).

Install transformers to be able to download and have access to HuggingFace's TinyLlama and LaMini models using the AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained() functions.

Install torch to be able to turn inputs into PyTorch tensors and reduce memory usage.

Both of the above libraries can be installed using "pip install transformers torch".

Alternatively, you can also recreate my environment using my "requirements.yaml" file
