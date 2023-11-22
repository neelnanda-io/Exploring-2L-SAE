# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gelu-2l")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
evals.sanity_check(model)
# %%
import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd

# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version, device_override=None):
        """
        Loads the saved autoencoder from HuggingFace. 
        
        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47
        
        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        if device_override is not None:
            cfg["device"] = device_override

        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self
encoder0 = AutoEncoder.load_from_hf("gelu-2l_L0_16384_mlp_out_51", "cuda")
encoder1 = AutoEncoder.load_from_hf("gelu-2l_L1_16384_mlp_out_50", "cuda")
# %%
data = load_dataset("NeelNanda/c4-10k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data[0]
# %%
example_tokens = tokenized_data[:200]["tokens"]
logits, cache = model.run_with_cache(example_tokens)
per_token_loss = model.loss_fn(logits, example_tokens, True)
imshow(per_token_loss)
# %%
original_mlp_out = cache["mlp_out", 1]
loss, reconstr_mlp_out, hidden_acts, l2_loss, l1_loss = encoder1(original_mlp_out)
def reconstr_hook(mlp_out, hook, new_mlp_out):
    return new_mlp_out
def zero_abl_hook(mlp_out, hook):
    return torch.zeros_like(mlp_out)
print("reconstr", model.run_with_hooks(example_tokens, fwd_hooks=[(utils.get_act_name("mlp_out", 1), partial(reconstr_hook, new_mlp_out=reconstr_mlp_out))], return_type="loss"))
print("Orig", model(example_tokens, return_type="loss"))
print("Zero", model.run_with_hooks(example_tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("mlp_out", 1), zero_abl_hook)]))
# %%

original_mlp_out = cache["mlp_out", 0]
loss, reconstr_mlp_out, hidden_acts, l2_loss, l1_loss = encoder0(original_mlp_out)
def reconstr_hook(mlp_out, hook, new_mlp_out):
    return new_mlp_out
def zero_abl_hook(mlp_out, hook):
    return torch.zeros_like(mlp_out)
print("reconstr", model.run_with_hooks(example_tokens, fwd_hooks=[(utils.get_act_name("mlp_out", 0), partial(reconstr_hook, new_mlp_out=reconstr_mlp_out))], return_type="loss"))
print("Orig", model(example_tokens, return_type="loss"))
print("Zero", model.run_with_hooks(example_tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("mlp_out", 0), zero_abl_hook)]))
# %%
orig_logits = model(example_tokens)
orig_ptl = model.loss_fn(orig_logits, example_tokens, True)

zero_logits = model.run_with_hooks(example_tokens, return_type="logits", fwd_hooks=[(utils.get_act_name("mlp_out", 0), zero_abl_hook)])
zero_ptl = model.loss_fn(zero_logits, example_tokens, True)

recons_logits = model.run_with_hooks(example_tokens, fwd_hooks=[(utils.get_act_name("mlp_out", 0), partial(reconstr_hook, new_mlp_out=reconstr_mlp_out))], return_type="logits")
recons_ptl = model.loss_fn(recons_logits, example_tokens, True)
# %%
histogram(recons_ptl.flatten())
# %%
scatter(x=(recons_ptl-orig_ptl).flatten(), y=(zero_ptl-orig_ptl).flatten())
delta_ptl = recons_ptl - orig_ptl
histogram(delta_ptl.flatten(), marginal="box")
# %%
scipy.stats.kurtosis(to_numpy(delta_ptl).flatten())
# %%
token_df = nutils.make_token_df(example_tokens).query("pos>=1")
token_df["delta_ptl"] = to_numpy(delta_ptl.flatten())

# %%
display(token_df.sort_values("delta_ptl", ascending=False).head(20))
display(token_df.sort_values("delta_ptl", ascending=True).head(20))
# %%
virtual_weights = encoder0.W_dec @ model.W_in[1] @ model.W_out[1] @ encoder1.W_enc
virtual_weights.shape
# %%
histogram(virtual_weights.flatten()[::1001])
# %%
neuron2neuron = model.W_out[0] @ model.W_in[1]
histogram(neuron2neuron.flatten()[::101])
# %%
histogram(virtual_weights.mean(0), title="Ave by end feature")
histogram(virtual_weights.mean(1), title="Ave by start feature")
histogram(virtual_weights.median(0).values, title="Median by end feature")
histogram(virtual_weights.median(1).values, title="Median by start feature")
# %%
example_tokens = tokenized_data[:800]["tokens"]
_, cache = model.run_with_cache(example_tokens, stop_at_layer=2, names_filter=lambda x: "mlp_out" in x)
loss, recons_mlp_out0, hidden_acts0, l2_loss, l1_loss = encoder0(cache["mlp_out", 0])
loss, recons_mlp_out1, hidden_acts1, l2_loss, l1_loss = encoder1(cache["mlp_out", 1])


# %%
try:
    hidden_acts0 = einops.rearrange(hidden_acts0, "batch pos d_enc -> (batch pos) d_enc")
    hidden_acts1 = einops.rearrange(hidden_acts1, "batch pos d_enc -> (batch pos) d_enc")
except:
    pass
hidden_is_pos0 = hidden_acts0 > 0
hidden_is_pos1 = hidden_acts1 > 0
d_enc = hidden_acts0.shape[-1]
cooccur_count = torch.zeros((d_enc, d_enc), device="cuda", dtype=torch.float32)
for end_i in tqdm.trange(d_enc):
    cooccur_count[:, end_i] = hidden_is_pos0[hidden_is_pos1[:, end_i]].float().sum(0)
# %%
num_firings0 = hidden_is_pos0.sum(0)
num_firings1 = hidden_is_pos1.sum(0)
cooccur_freq = cooccur_count / torch.maximum(num_firings0[:, None], num_firings1[None, :])
# %%
# cooccur_count = cooccur_count.float() / hidden_acts0.shape[0]
# %%
histogram(cooccur_freq[cooccur_freq>0.1], log_y=True)
# %%
