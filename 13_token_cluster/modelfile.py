from tinygrad import Tensor, nn, Variable # type: ignore
from extra.models.llama import TransformerBlock, Attention, precompute_freqs_cis, apply_rotary_emb, repeat_kv # type: ignore

from typing import List, Dict, Union, Optional, Tuple
from sentencepiece import SentencePieceProcessor # type: ignore
from tqdm import trange # type: ignore
import math

TOKEN_DIMS   = 256
CLUSTER_SIZE = 8
CLUSTER_DIMS = 1024

MAX_CLUSTER_CONTEXT = 64

NORM_EPS = 1e-5

def __call__(self:Attention, x:Tensor, start_pos, freqs_cis:Tensor, mask:Optional[Tensor]) -> Tensor:
   xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

   xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
   xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
   xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

   xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
   bsz, seqlen, _, _ = xq.shape

   keys = xk
   values = xv

   keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
   xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
   attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
   attn = attn.reshape(bsz, seqlen, -1)
   return self.wo(attn)
Attention.__call__ = __call__

class Encoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      self.tok_embeddings = nn.Embedding(vocab_size, token_dim)
      self.cluster_embed = nn.Linear(self.model_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, int(self.model_dim*ff_mult), self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.out_proj = nn.Linear(self.model_dim, cluster_dim)
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
   
   def __call__(self, tokens:Tensor) -> Tensor:
      h = self.tok_embeddings(tokens)

      B, T, _ = h.shape
      assert T % self.cluster_size == 0
      C = T // self.cluster_size
      x = self.cluster_embed(h.reshape(B, C, self.model_dim))

      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
      return self.out_proj(x)

class Decoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size
      
      self.in_proj = nn.Linear(cluster_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, int(self.model_dim*ff_mult), self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
      self.output = nn.Linear(token_dim, vocab_size, bias=False)
   
   def __call__(self, c:Tensor) -> Tensor:
      x = self.in_proj(c)
      B, C, _ = x.shape
      T = C * self.cluster_size

      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
   
      logits = self.output(x.reshape(B, T, self.model_dim // self.cluster_size))
      return logits

class Generator:
   def __init__(self, max_context:int, n_layers:int, dim:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.layers = [TransformerBlock(dim, int(dim*ff_mult), dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
   
   def __call__(self, x:Tensor) -> Tensor:
      C = x.shape[1]
      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
      return x

class CombinedModel:
   tokenizer: Optional[SentencePieceProcessor] = None

   def __init__(self, vocab_size:int, max_context:int, n_layers:Dict[str,int], token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mults:Dict[str,float]):
      self.enc = Encoder  (vocab_size, max_context,   n_layers["enc"], token_dim, cluster_dim, cluster_size, d_head, ff_mults["enc"])
      self.gen = Generator(            max_context-1, n_layers["gen"],            cluster_dim,               d_head, ff_mults["gen"])
      self.dec = Decoder  (vocab_size, max_context,   n_layers["dec"], token_dim, cluster_dim, cluster_size, d_head, ff_mults["dec"])

      self.vocab_size = vocab_size
      self.max_context = max_context
      self.cluster_size = cluster_size

   def get_trainable_parameters(self, predict_loss:bool, decoded_loss:bool, cluster_loss:bool) -> List[Tensor]:
      params: List[Tensor] = []
      if decoded_loss or predict_loss:
         params += nn.state.get_parameters(self.enc)
      if cluster_loss or predict_loss:
         params += nn.state.get_parameters(self.gen)
      if decoded_loss or predict_loss:
         params += nn.state.get_parameters(self.dec)
      return params

   def compute_loss(self, orig_tokens:Tensor, predict_loss:bool, decoded_loss:bool, cluster_loss:bool) -> Tuple[Dict[str,Tensor],Tensor]:
      enc_clusters = self.enc(orig_tokens)

      losses: Dict[str,Tensor] = {}
      if predict_loss:
         prd_clusters = self.gen(enc_clusters[:, :-1])
         prd_tokens   = self.dec(prd_clusters)
         losses["predict"] = prd_tokens.sparse_categorical_crossentropy(orig_tokens[:, CLUSTER_SIZE:]).realize()
      if decoded_loss:
         dec_tokens   = self.dec(enc_clusters)
         losses["decoded"] = dec_tokens.sparse_categorical_crossentropy(orig_tokens).realize()
      if cluster_loss:
         prd_clusters = self.gen(enc_clusters[:, :-1].detach())
         losses["cluster"] = (enc_clusters.detach()[:, 1:] - prd_clusters).square().mean().mul(1000).log().realize()

      if   predict_loss: acc = (prd_tokens.argmax(axis=-1) == orig_tokens[:, CLUSTER_SIZE:]).mean().realize()
      elif decoded_loss: acc = (dec_tokens.argmax(axis=-1) == orig_tokens                  ).mean().realize()
      else:              acc = Tensor.zeros(1) - 0.01

      return losses, acc

   def generate(self, text_init:str, amount:int) -> str:
      if self.tokenizer is None:
         self.tokenizer = SentencePieceProcessor(model_file="/raid/downloads/LLaMA-2/7B/tokenizer.model")
      assert self.tokenizer is not None

      tokens = self.tokenizer.Encode(text_init)
      spare = len(tokens) % self.cluster_size
      if spare > 0:
         tokens = tokens[:-spare]
      assert len(tokens) > 0, f"got no remaining tokens after cutting off {spare} spare"
      assert len(tokens) % self.cluster_size == 0, "FATAL: invalid computation"

      gen_amount = math.ceil(amount / self.cluster_size)

      x = Tensor(tokens).unsqueeze(0)
      z = self.enc(x)
      for _ in trange(gen_amount):
         z_h = self.gen(z)
         z = z.cat(z_h[:, -1:, :], dim=1).realize()
      tokens = self.dec(z).argmax(-1).numpy().tolist()

      return self.tokenizer.Decode(tokens)[0]

def create_models():
   MODEL_CONFIGS = 3

   VOCAB_SIZE = 32000
   D_HEAD = 32

   # Define Models
   layers = {
      "enc": 4,
      "gen": 32,
      "dec": 4,
   }
   ff_mults = {
      "enc": 1.5,
      "gen": 3.0,
      "dec": 1.5,
   }
   return [CombinedModel(VOCAB_SIZE, MAX_CLUSTER_CONTEXT, layers, TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD, ff_mults) for _ in range(MODEL_CONFIGS)]
