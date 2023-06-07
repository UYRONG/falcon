from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

def generate(input_text,rrmodel,tokenizer):
    print("----"*10)
    print("Input:")
    print(input_text)
    print("----"*10)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to('cuda')

    attention_mask = torch.ones(input_ids.shape)

    output = rrmodel.generate(input_ids, 
                attention_mask=attention_mask, 
                max_length=800,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Output:")
    print(output_text)
    print("----"*10)


def main():
    start_time = time.time()
    model = "tiiuae/falcon-7b"

    rrmodel = AutoModelForCausalLM.from_pretrained(model, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        offload_folder="offload",
        cache_dir="/scratch1/yrong016")

    tokenizer = AutoTokenizer.from_pretrained(model)

    file1 = open('prompt.txt', 'r')
    Lines = file1.readlines()[0:1]

    count = 0

    for line in Lines:
        count += 1
        input_text = line.strip()
        print("EXAMPLE " + str(count) + ":")
        generate(input_text,rrmodel,tokenizer)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")

if __name__ == "__main__":
    main()