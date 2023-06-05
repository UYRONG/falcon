from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b"

rrmodel = AutoModelForCausalLM.from_pretrained(model, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    offload_folder="offload")

tokenizer = AutoTokenizer.from_pretrained(model)


print("----"*10)
print("Input:")
# input_text = "Tell me the dialogue action annotation of the utterance 'i want the books and you take the rest' from the following options: propose, agree, greet "

input_text = "utterance: YOU: i would like 4 hats and you can have the rest . <eos> annotation: propose book = 0 hat = 4 ball = 0 utterance: THEM: if i can have the hats and the books you can have the ball <eos>  annotation: propose book = 0 hat = 4 ball = 0 utterance: YOU: so you want me to only have the ball ? no deal make me a better offer <eos>  annotation: propose book = 0 hat = 4 ball = 0 utterance: THEM: i will take the hats and two books and you can have the rest <eos> annotation: "


print(input_text)
print("----"*10)
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
print("Output:")
print(output_text)
print("----"*10)