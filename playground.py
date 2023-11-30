from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Linear # type: ignore
from tinygrad.helpers import dtypes, all_int # type: ignore
from typing import Optional # type: ignore
import math




def test_batch_causal_masks():

   def scaled_dot_product_attention(self:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
      # NOTE: it works if key, value have symbolic shape
      assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
      if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
      if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
      if attn_mask is not None:
         print("\nAttention mask right before applying")
         print(attn_mask.numpy())
      a = self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])
      b = a + attn_mask
      print("\nOriginal Query @ Key results")
      print(a.shape)
      print(a.numpy())
      print("\nMasked results")
      print(b.shape)
      print(b.numpy())
      return b.softmax(-1).dropout(dropout_p) @ value

   class CrossAttention:
      def __init__(self, query_dim, context_dim, n_heads, d_head, dropout=0.1, is_causal=False):
         self.to_q = Linear(query_dim,   n_heads*d_head, bias=False)
         self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
         self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
         self.num_heads = n_heads
         self.head_size = d_head
         self.to_out = [Linear(n_heads*d_head, query_dim)]
         self.dropout = dropout
         self.is_causal = is_causal
      def __call__(self, x:Tensor, context:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
         context = x if context is None else context
         q,k,v = self.to_q(x), self.to_k(context).expand((x.shape[0],*context.shape[1:])), self.to_v(context).expand((x.shape[0],*context.shape[1:]))
         q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
         if attn_mask is not None: attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, *attn_mask.shape[1:]).expand((attn_mask.shape[0], self.num_heads, *attn_mask.shape[1:]))
         attention = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal).dropout(self.dropout).transpose(1,2)
         h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
         return h_.sequential(self.to_out)

   Tensor.manual_seed(1337)

   BS = 2
   dim = 8
   ctx = 5
   den = 3
   n_heads = 2

   # BS  = 5
   # dim = 256
   # ctx = 256
   # den = 8
   # n_heads = 8
   to_b = lambda x: x.cast(dtypes.bool) if True else x

   ca = CrossAttention(dim, dim, n_heads, dim//n_heads)

   context = Tensor.arange(ctx*dim).reshape(1,1,ctx,dim).expand(BS,ctx,ctx,dim).reshape(-1,ctx,dim)
   inputs  = Tensor.ones(BS,ctx,den,dim).reshape(-1,den,dim)
   attn_mask = to_b(Tensor.ones(ctx,ctx).tril(0)).reshape(1,ctx,1,ctx).expand(BS,ctx,den,ctx).reshape(-1,den,ctx)
   print("\nOriginal attention mask")
   print(attn_mask.shape)
   print(attn_mask.numpy())

   ca(inputs, context, attn_mask=attn_mask)
   # ca(inputs, context)


def test_builtin_causal_mask():
   def scaled_dot_product_attention(self:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
      # NOTE: it works if key, value have symbolic shape
      assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
      if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
      if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
      if attn_mask is not None: print(attn_mask.numpy())
      return (self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value
   
   x = Tensor.ones(1,1,4,4)
   scaled_dot_product_attention(x, x, x, is_causal=True)

def softmax_tests():
   Tensor.manual_seed(1337)
   x = Tensor.randn(4,4)
   # print(x.softmax().numpy())
   y = x + Tensor.ones(*x.shape)
   # print(y.softmax().numpy())
   diff = x.softmax() - y.softmax()
   print(diff.numpy())


if __name__ == "__main__":
   test_batch_causal_masks()
   # test_builtin_causal_mask()
   # softmax_tests()
