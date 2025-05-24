from sentence_transformers import SentenceTransformer, util # type: ignore

def main():
   sentences = ["I'm happy", "I'm full of happiness"]

   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

   # Compute embedding for both lists
   embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
   embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

   sim = util.pytorch_cos_sim(embedding_1, embedding_2)
   print(sim)

if __name__ == "__main__":
   main()
