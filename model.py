import torch
from machine_comprehension.models import MatchLSTM


class MatchLSTMModified(torch.nn.Module):
    def __init__(self, network_input_size, match_lstm_input_size, word_embedding=False, part_of_speech=False, knowledge_graph=False):
        super().__init__()
        self.reduce_embedding = torch.nn.Linear(network_input_size, match_lstm_input_size)
        self.match_lstm = MatchLSTM(match_lstm_input_size)
        self.embedding = Embedding(word_embedding=word_embedding, part_of_speech=part_of_speech, knowledge_graph=knowledge_graph)


    def forward(self, word_context, word_question, kg_context, kg_question, pos_context, pos_question, context_lengths, question_lengths):

        # Este reduce hay que modificarlo seguro.
        # Hay que hacer padding y esas cosas para que se aplique la reducción a cada palabra, no a la frase completa.
        context_vec, question_vec = self.embedding(word_context, word_question, kg_context, kg_question, pos_context, pos_question)

        return self.match_lstm(context_vec, context_lengths, question_vec, question_lengths)


class Embedding(torch.nn.Module):
    def __init__(self, word_embedding=False, part_of_speech=False, knowledge_graph=False):
        super().__init__()
        self.embedding = {}

        if isinstance(word_embedding, torch.Tensor):
            self.embedding['word'] = torch.nn.Embedding.from_pretrained(word_embedding)
        if isinstance(part_of_speech, torch.Tensor):
            self.embedding['pos'] = torch.nn.Embedding.from_pretrained(part_of_speech)
        if isinstance(knowledge_graph, torch.Tensor):
            self.embedding['kg'] = torch.nn.Embedding.from_pretrained(knowledge_graph)

    def forward(self, word_context, word_question, kg_context, kg_question, pos_context, pos_question):
        context = self.apply_embedding(word_context, pos_context, kg_context)
        question = self.apply_embedding(word_question, pos_question, kg_question)

        # TODO check que el cat esté en la dimensión correcta
        return torch.cat(context, dim=0), torch.cat(question, dim=0)

    def apply_embedding(self, word, pos, kg):
        applied_embeddings = []
        if 'word' in self.embedding:
            applied_embeddings.append(self.embedding['word'](word))
        if 'pos' in self.embedding:
            applied_embeddings.append(self.embedding['pos'](pos))
        if 'kg' in self.embedding:
            applied_embeddings.append(self.embedding['kg'](kg))
        return applied_embeddings