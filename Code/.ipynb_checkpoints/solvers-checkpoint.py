from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from openai import OpenAI
import torch

class LLM:


    def __init__(self, model ="mistralai/Mistral-7B-v0.1", device = 'auto'):

        print("Loading model...")
        self.model_name = model

        if 'gpt' in model and 'community' not in self.model_name:
            self.model = OpenAI(api_key = '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')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map=device)
        print("Done.")


    def generate(self,text, max_length = 2000):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'gpt' in self.model_name and 'community' not in self.model_name:
            completion = self.model.chat.completions.create(
                          model=self.model_name,#"gpt-3.5-turbo-0613",
                          messages=text
                        )

            result_decoded = completion.choices[0].message.content

        else:
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        
            print("Tokenizing message")
                
            model_inputs = tokenizer.apply_chat_template(text, tokenize=False)
            model_inputs = tokenizer([model_inputs], return_tensors="pt").to(device)
            #print(model_inputs)
        
            print("Forward pass through model...")
            result = self.model.generate(**model_inputs, max_length = max_length)
            result_decoded = tokenizer.batch_decode(result, skip_special_tokens = True)[0]
            result_decoded = result_decoded.replace('<|im_start|>', "").replace('<|im_end|>', "").split('\n')
            
        return(result_decoded)

def __select_literal(cnf):
    for c in cnf:
        for literal in c:
            return literal[0]

def brute_force(cnf):
    literals = set()
    for conj in cnf:
        for disj in conj:
            literals.add(disj[0])
 
    literals = list(literals)
    n = len(literals)
    steps = 0
    for seq in itertools.product([True,False], repeat=n):
        steps += 1
        a = set(zip(literals, seq))
        if all([bool(disj.intersection(a)) for disj in cnf]):
            return True, a, steps
 
    return False, None, steps
 
def dpll(cnf, assignments={}, steps = 1):
    
    if len(cnf) == 0:
        return True, assignments, steps
 
    if any([len(c)==0 for c in cnf]):
        return False, None, steps
 
    l = __select_literal(cnf)
    
    new_cnf = [c for c in cnf if (l, True) not in c]
    new_cnf = [c.difference({(l, False)}) for c in new_cnf]
    sat, vals, steps = dpll(new_cnf, {**assignments, **{l: True}}, steps + 1)
    if sat:
        return sat, vals, steps
 
    new_cnf = [c for c in cnf if (l, False) not in c]
    new_cnf = [c.difference({(l, True)}) for c in new_cnf]
    sat, vals, steps = dpll(new_cnf, {**assignments, **{l: False}}, steps + 1)
    if sat:
        return sat, vals, steps
 
    return False, None, steps