import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Loads pretrained GPT 2 model from Hugging Face using transformers library
# Returns model and tokenizer objects for GPT 2
def load_gpt2():
    print("Loading GPT 2")
    mname = "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float32)
        print(f"Successfully loaded GPT 2")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading GPT 2: {e}")
        return None, None
    
# Loads pretrained GPT 2 Medium model from Hugging Face using transformers library
# Returns model and tokenizer objects for GPT 2 Medium
def load_gpt2m():
    print("Loading GPT 2 Medium")
    mname = "openai-community/gpt2-medium"
    try:
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float32)
        print(f"Successfully loaded GPT 2 Medium")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading GPT 2 Medium: {e}")
        return None, None

# Reads prompts from a text file, one prompt per line
# Takes a string of the file's name ("prompts.txt") as an argument
# Returns a list of the prompts as strings
def read_prompts(filename):
    with open(filename, "r") as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]

    return prompts

# Takes a single line from prompts.txt, model object, and the model's tokenizer as arguments
# Generates a response with a maximum length of 100
# Returns token IDs translated to a string that is readable by humans
def respond(prompt, model, tokenizer):
    # Returns input prompt as a torch.tensor object so it can be passed directly into a PyTorch model
    input = tokenizer(prompt, return_tensors = "pt")
    # Disables gradient calculation in PyTorch, reducing memory usage (one of my biggest obstacles)
    with torch.no_grad():
        output = model.generate(**input, max_length=100)

    return tokenizer.decode(output[0], skip_special_tokens = True)

# Writes each response from the model that the function is called for to the outputs.txt file
def write_to_file(responses, mname):
    with open("outputs.txt", "a") as file:
        file.write(f"---------RESPONSES FROM {mname}---------\n")
        for response in responses:
            file.write(response + "\n")
        file.write("\n")
    print("Responses saved to outputs.txt")

def main():
    print("Hello World")
    # Separately stores model object and tokenizer for both GPT 2 and GPT 2 Medium after loading the 2 models
    gpt2_model, gpt2_tokenizer = load_gpt2()
    gpt2m_model, gpt2m_tokenizer = load_gpt2m()
    
    # Returns prompts as a list of the lines from prompts.txt, has GPT 2 and GPT 2 Medium respond separately, 
    # then appends GPT 2 and GPT 2Medium responses to their respective lists
    fname = "prompts.txt"
    gpt2_responses = []
    gpt2m_responses = []
    prompts = read_prompts(fname)
    for prompt in prompts:
        gpt2_response = respond(prompt, gpt2_model, gpt2_tokenizer)
        gpt2_responses.append(gpt2_response)
        gpt2m_response = respond(prompt, gpt2m_model, gpt2m_tokenizer)
        gpt2m_responses.append(gpt2m_response)
    
    #Writes GPT 2 then GPT 2 Medium responses to outputs.txt file
    write_to_file(gpt2_responses, "GPT 2")
    write_to_file(gpt2m_responses, "GPT 2 Medium")

if __name__ == "__main__":
    main()