# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:38:09 2023

source: https://huggingface.co/spaces/tloen/alpaca-lora/blob/main/app.py
online demo: https://huggingface.co/spaces/tloen/alpaca-lora

The script will download and use the LLAMA model with weights provided from Alpaca-LoRA (Low-Rank LLaMA Instruct-Tuning).
The chatbot will run locally on your computer.

PEFT stands for State-of-the-art Parameter-Efficient Fine-Tuning (PEFT).
pip install peft

You may need to update your version of Transformers to the latest.

The queries are composed of two parameters: "instruction" and "input" (optional)
Example of a prompt that uses both parameters: 
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}"

CPU
====
Processing on a CPU takes a lot of time for a single text generation! It takes between 20 minutes and 1H on a regular 
laptop in CPU mode. 10GB of RAM are used. You may need to restart yout Python kernel if you are using Spyder.

GPU
====
This model runs OK in GPU mode on Google Colab with 16GB GPU.

It seems there are problems when running on Windows with CUDA with limited amount of memory of GPU memory (4GB):
    - offload_folder problem - FIXED in latest github version
    - model.half() - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    - "ValueError: weight is on the meta device, we need a value to put in on cpu" in modeling.py
"""

import torch
from peft import PeftModel
import transformers
import os, time

assert ("LlamaTokenizer" in transformers._import_structure["models.llama"]), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"

force_cpu = False

if torch.cuda.is_available() and not force_cpu:
    device = "cuda"
    print("Video memory available:", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, "MBs")
else:
    device = "cpu"
print("Compute device is:", device)

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

print("Loading model with selected weights ...")

if device == "cuda":
    
    print("model on cuda")
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload", #required on GPU with not enough memory
    )
    
    model = PeftModel.from_pretrained(
        model, 
        LORA_WEIGHTS, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload", #required on GPU with not enough memory
        
    )#.to("cuda")
elif device == "mps": # Metal Performance Shaders (MPS) backend for GPU training acceleration for Mac computers with Apple silicon or AMD GPUs
   
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else: #CPU

    print("model on CPU")
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, 
        device_map={"": device}, 
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
    
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

#half() generates errors when not used on a large 16 GB GPU
if device != "cpu": #half() is not available for CPU
     model.half()
    
model.eval()

if torch.__version__ >= "2" and not os.name == 'nt':
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    
    print("Generating ...")
    start = time.time()
    
    with torch.no_grad():
        generation_output = model.generate( #Generates sequences of token ids for models with a language modeling head.
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    
    s = generation_output.sequences[0]
    
    output = tokenizer.decode(s)
    
    end = time.time()
    print('Response generation time:', (end - start) / 60, 'minutes')
    
    return output.split("### Response:")[1].strip()

if __name__ == "__main__":

    #examples
    for instruction in [
        #"Tell me about alpacas.",
        #"Tell me about the president of Mexico in 2019.",
        # "Tell me about the king of France in 2019.",
         "List all Canadian provinces in alphabetical order.",
        # "Write a Python program that prints the first 10 Fibonacci numbers.",
        # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        # "Tell me five words that rhyme with 'shock'.",
        # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        # "Count up from 1 to 500.",
    ]:
        print("---------------------------")
        
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        
        print()