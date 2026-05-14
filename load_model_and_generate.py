from transformers import AutoModelForCausalLM, AutoTokenizer

# load the model first
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", # model repo - Here I'm using a low-parameters model from Alibaba
    device_map = "cuda",
    torch_dtype = "auto", 
    trust_remote_code = False
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


#prompt generation - I'm not adding <|assistant|> manually here as Qwen3 inserts its own assistant turn token during generation
prompt = "Explain in short, what quantum computing is. <think> </think>" # Adding <think> </think> to skip Qwen3 reasoning.

# Convert the prompt, into tokens
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

#Optional, for troubleshooting purposes
# print(f"Input tokens are: {input_ids}")

# Generate the output
output_generation = model.generate(
    input_ids = input_ids,
    max_new_tokens = 200
)

# Print the decoded tokens
print(tokenizer.decode(output_generation[0]))
