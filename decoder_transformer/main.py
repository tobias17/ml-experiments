from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
import yaml, os

class TransformerBlock:
   def __init__(self, embed_dim, num_heads, ff_dim, act=lambda x: x.relu(), dropout=0.1):
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

      self.num_heads = num_heads
      self.head_size = embed_dim // num_heads
      self.act = act
      self.dropout = dropout

      self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.key   = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
      self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
      self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

   def attn(self, x:Tensor) -> Tensor:
      # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
      query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
      attention = Tensor.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(1,2)
      return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

   def __call__(self, x:Tensor) -> Tensor:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
      return x

class Transformer:
   def __init__(self, vocab_size: int, max_context: int, layers: int, embed_dim: int, n_heads: int, ff_dim: int):
      self.tok_embed = Embedding(vocab_size,  embed_dim)
      self.pos_embed = Embedding(max_context, embed_dim)
      self.layers = [TransformerBlock(embed_dim, n_heads, ff_dim) for _ in range(layers)]
      self.class_head = Linear(embed_dim, vocab_size)

def train(model: Transformer):
   pass

if __name__ == "__main__":
   yaml_filepath = os.path.join(os.path.dirname(__file__), "config.yaml")
   with open(yaml_filepath) as f:
      yaml_data = yaml.safe_load(f)

   model = Transformer(**yaml_data["model_params"])
   train(model)
