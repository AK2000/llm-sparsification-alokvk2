import torch
import torch.nn.utils.prune as prune
from transformers import GPT2Model, T5Model, DebertaV2Model

def prune_model(model, amount=0.2):
    def flatten(m):
        f = [x for c in m.children() for x in flatten(c)]
        if len(f) == 0:
            return [m]
        else:
            return f
    layers = flatten(model)
    params_to_prune = [(m, p[0]) for m in layers for p in m.named_parameters()]
    
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model


model_dir = "models/"
for sparsity in [.1, .5, .9, .95, .99]:
    model = GPT2Model.from_pretrained("gpt2")
    model = prune_model(model, amount=sparsity)
    name = f"gpt2-{sparsity}.pt"
    model.save_pretrained(model_dir + name)

for sparsity in [.1, .5, .9, .95, .99]:
    model = T5Model.from_pretrained("t5-large")
    model = prune_model(model, amount=sparsity)
    name = f"t5-{sparsity}.pt"
    model.save_pretrained(model_dir + name)

for sparsity in [.1, .5, .9, .95, .99]:
    model = DebertaV2Model.from_pretrained("microsoft/deberta-v2-xxlarge")
    model = prune_model(model, amount=sparsity)
    name = f"deberta-{sparsity}.pt"
    model.save_pretrained(model_dir + name)