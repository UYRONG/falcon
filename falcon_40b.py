from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

def generate(input_text, tokenizer, rrmodel):
    print("----"*10)
    print("Input:")
    print(input_text)
    print("----"*10)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to('cuda')

    attention_mask = torch.ones(input_ids.shape)

    output = rrmodel.generate(input_ids, 
                attention_mask=attention_mask, 
                max_length=300,
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
    model = "tiiuae/falcon-40b"

    rrmodel = AutoModelForCausalLM.from_pretrained(model, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        offload_folder="offload",
        cache_dir="./hf_cache")

    print('*'*50)
    print(rrmodel.hf_device_map)
    print('*'*50)

    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]

    count = 0

    for line in prompts:
        count += 1
        input_text = line.strip()
        print("EXAMPLE " + str(count) + ":")
        generate(input_text, tokenizer, rrmodel)

if __name__ == "__main__":
    main()