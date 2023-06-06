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

print("----"*10)
print("Input:")
# input_text = "utterance: YOU: i would like 4 hats and you can have the rest . <eos> annotation: propose book = 0 hat = 4 ball = 0 utterance: THEM: if i can have the hats and the books you can have the ball <eos>  annotation: propose book = 0 hat = 4 ball = 0 utterance: YOU: so you want me to only have the ball ? no deal make me a better offer <eos>  annotation: propose book = 0 hat = 4 ball = 0 utterance: THEM: i will take the hats and two books and you can have the rest <eos> annotation: "
input_text = "utterance: YOU: i'll take the hat , you can have the rest .  <eos> annotation: propose book=0 hat=1 ball=0 utterance: THEM: i would like the balls , hat and 2 books  <eos> annotation:  propose book = 2 hat = 0 ball = 0 utterance: YOU: i can give you the hat and the balls if i keep the books .  <eos> annotation: propose book = 4 hat = 0 ball = 0 utterance: THEM: sorry that wont work . <eos> annotation: disagree utterance: YOU: can you either give me the hat or 3 of the books ? <eos> annotation:  propose book = 4 hat = 1 ball = 0 utterance: THEM: sorry looks like we wont be making a deal <eos> annotation: disagree  utterance: YOU: yeah i have no idea how to do that .<eos> annotation: disagree utterance: THEM: we keep saying no deal until the no deal button appears <eos> annotation: disagree utterance: YOU: oh , how about i get 2 books ? <eos>  annotation: propose book = 2 hat = 0 ball = 0 utterance: THEM: you get 2 books and i get the rest ? <eos> annotation:"
print(input_text)
print("----"*10)

input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones(input_ids.shape)

output = rrmodel.generate(input_ids, 
            attention_mask=attention_mask, 
            max_length=300,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Output:")
print(output_text)
print("----"*10)