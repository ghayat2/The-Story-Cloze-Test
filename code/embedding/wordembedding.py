
import gensim
import numpy as np

class word_embedding:

    def __init__(self,path,vocab,vocab_size,embedding_dimension):
        self.path = path
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.dim_ebding = embedding_dimension

    def embedding(self):
         gensim.model = model.KeyedVectors.load_word2vec_format(self.path,binary = False)
         external_embedding = np.zeros(shape = (self.vocab_size,self.dim_ebding))

         for tok, idx in self.vocab.items():
             if tok in model.vocab:
                 external_embedding[idx] = model[tok]
        ### in this case, the word not in dictionary will be 0 vector of size dim_ebding

         return external_embedding


