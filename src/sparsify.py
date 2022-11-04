import torch
import torch.nn.utils.prune as prune
from transformers import GPT2Tokenizer, GPT2Model, RobertaModel, PegasusModel

def prune_model(model, amount=0.2):
    def flatten(m):
        f = [x for c in m.children() for x in flatten(c)]
        if len(f) == 0:
            return [m]
        else:
            return f
    layers = flatten(model)
    params_to_prune = [(m, p[0]) for m in layers for p in m.named_parameters() if p[0][-6:] == "weight"]
    print(params_to_prune)

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for (m, name) in params_to_prune:
        try:
            prune.remove(m, name)
        except Exception as e:
            print(e)

    return model

model_dir = "models/"

# Add padding token to tokenizer for GPT2 for seq classification task
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(model_dir + "gpt2_tokenizer")

for sparsity in [.1, .5, .9, .95, .99]:
   model = GPT2Model.from_pretrained("gpt2")
   model = prune_model(model, amount=sparsity)
   model.config.pad_token_id = model.config.eos_token_id
   name = f"gpt2-{sparsity}"
   model.save_pretrained(model_dir + name)

# Save a gpt2 model with padding token for seq classification task
model = GPT2Model.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
name = f"gpt2-{0.0}"
model.save_pretrained(model_dir + name)

for sparsity in [.1, .5, .9, .95, .99]:
    model = RobertaModel.from_pretrained("roberta-large")
    model = prune_model(model, amount=sparsity)
    name = f"roberta-{sparsity}"
    model.save_pretrained(model_dir + name)

for sparsity in [.1, .5, .9, .95, .99]:
    model = PegasusModel.from_pretrained("google/pegasus-large")
    model = prune_model(model, amount=sparsity)
    name = f"deberta-{sparsity}.pt"
    model.save_pretrained(model_dir + name)


