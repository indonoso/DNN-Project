import torch
from machine_comprehension.models import MatchLSTM
from utils import load_embeddings


class MatchLSTMModified(torch.nn.Module):
    def __init__(self, network_input_size, match_lstm_input_size, word_embedding=False, part_of_speech=False, knowledge_graph=False):
        super().__init__()
        self.reduce_embedding = torch.nn.Linear(network_input_size, match_lstm_input_size)
        self.match_lstm = MatchLSTM(match_lstm_input_size)
        self.embedding = Embedding(word_embedding=word_embedding, part_of_speech=part_of_speech, knowledge_graph=knowledge_graph)


    def forward(self, context, context_lengths, question, question_lengths):

        # Este reduce hay que modificarlo seguro.
        # Hay que hacer padding y esas cosas para que se aplique la reducción a cada palabra, no a la frase completa.
        context_vec = self.reduce_embedding(self.embedding(context))
        question_vec = self.reduce_embedding(self.embedding(question))

        return self.match_lstm(context_vec, context_lengths, question_vec, question_lengths)


class Embedding(torch.nn.Module):
    def __init__(self, word_embedding=False, part_of_speech=False, knowledge_graph=False):
        super().__init__()
        self.embedding = []

        if word_embedding:
           self.embedding.append(self.load_embeding_layer('word_embedding', word_embedding))
        if part_of_speech:
            self.embedding.append(self.load_embeding_layer('part_of_speech', part_of_speech))
        if knowledge_graph:
            self.embedding.append(self.load_embeding_layer('knwoledge_graph', knowledge_graph))

    def load_embeding_layer(self, kind, size):
        weights = load_embeddings(kind, size)
        return torch.nn.Embedding.from_pretrained(weights)

    def forward(self, context, question):
        # Si hay que hacer alguna modificación de los emebddings o algo aquí va
        # Creo que mi versión está demasiado simplicada. context y question podrían ser unos dicts con los token ids para cada representación
        # En ese caso habría que cambiar esto para que funcione. Se puede hacer con dicts

        context_embedding = [emb(context) for emb in self.embedding]
        question_embedding = [emb(question) for emb in self.embedding]
        # TODO check que el cat esté en la dimensión correcta
        return torch.cat(context_embedding, dim=0), torch.cat(question_embedding, dim=0)


