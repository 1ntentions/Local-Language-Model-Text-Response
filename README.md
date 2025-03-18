# Local-Language-Model-Project
In this project 2 local language models, openai-community/gpt2 and openai-community/gpt2-medium, are loaded from HuggingFace.

GPT 2 has 137 million parameters, while GPT 2 Medium has 380 million parameters.

## Steps
-Have python 3.8 or higher installed (I am personally using 3.12).

-The torch library is necessary to turn inputs into PyTorch tensors and reduce memory usage.

-The transformers library is necessary to download HuggingFace's GPT 2 and GPT 2 Medium models and tokenizers for them using the AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained() functions.

-Add the torch and transformers libraries into your program with the import statements "import torch" and "from transformers import AutoModelForCausalLM, AutoTokenizer".

-You can recreate my environment using my "requirements.yaml" file
