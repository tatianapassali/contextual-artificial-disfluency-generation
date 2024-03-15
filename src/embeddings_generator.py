from transformers import AutoTokenizer, AutoModel
import transformers
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print(transformers.__version__)

class EmbeddingGenerator:
    def __init__(self):
        self.tokenizer = self._load_tokenizer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model = self._load_model("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    def _load_model(self, model_path):
        return AutoModel.from_pretrained(model_path)

    def _load_tokenizer(self, tokenizer_path):
        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        # print(token_embeddings.shape, token_embeddings.size())
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _generate_bert_embeddings(self, sentence):
        encoded_input = self.tokenizer.encode_plus(sentence,
                                                   max_length=100,
                                                   padding=True,
                                                   truncation=True,
                                                   truncation_strategy='max_length',
                                                   return_tensors='pt')

        # Get the encoded output
        model_output = self.model(**encoded_input)

        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask']).detach().numpy()

        return embeddings

    def _compute_cosine_similarity(self, sentence, candidate_sentence):
        sentence_embeddings = self._generate_bert_embeddings(sentence)
        candidate_sentence_embeddings = self._generate_bert_embeddings(candidate_sentence)

        cosine_similarity_score = cosine_similarity(sentence_embeddings, candidate_sentence_embeddings)

        return cosine_similarity_score

    def _create_sentence_candidates(self, tokens, idx, word_candidates):
        sentence_candidates = []
        for word in word_candidates:
            tokens[idx] = word
            sentence_candidates.append(" ".join(tokens))

        return sentence_candidates

    def select_candidate(self, tokens, idx, word_candidates):
        original_sentence = " ".join(tokens)
        sentence_candidates = self._create_sentence_candidates(tokens, idx, word_candidates)
        similarities = []
        for cand_sentence in sentence_candidates:
            sim = self._compute_cosine_similarity(original_sentence, cand_sentence)
            similarities.append(sim)

        if len(similarities) > 0:
            max_sim_idx = np.argmax(similarities)
            return word_candidates[max_sim_idx]
        else:
            return None
