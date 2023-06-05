from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b"

rrmodel = AutoModelForCausalLM.from_pretrained(model, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    offload_folder="offload",
    cache_dir="/scratch1/yrong016")

tokenizer = AutoTokenizer.from_pretrained(model)

input_text = "predict the dialogue act of the utterance 'i want the books and you take the rest' "
input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones(input_ids.shape)

output = rrmodel.generate(input_ids, 
            attention_mask=attention_mask, 
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)