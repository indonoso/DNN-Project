import torch
from machine_comprehension.models import MatchLSTM


class MatchLSTMModified(torch.nn.Module):
    def __init__(self, device, network_input_size, match_lstm_input_size, word_embedding=False, part_of_speech=False,
                 knowledge_graph=False, hidden_size=150):
        super().__init__()
        self.embedding = Embedding(device, word_embedding=word_embedding,
                                   part_of_speech=part_of_speech,
                                   knowledge_graph=knowledge_graph)
        self.reduce_embedding = torch.nn.Linear(network_input_size, match_lstm_input_size)
        self.match_lstm = MatchLSTM(embedding_size=match_lstm_input_size, hidden_size=hidden_size)

    def forward(self, word_context, word_question, kg_context, kg_question, pos_context, pos_question,
                context_lengths, question_lengths):

        context_vec, question_vec = self.embedding(word_context, word_question, kg_context, kg_question,
                                                   pos_context, pos_question)

        context_vec = self.reduce_embedding(context_vec)
        question_vec = self.reduce_embedding(question_vec)

        question_vec = torch.transpose(question_vec, 0, 1)
        context_vec = torch.transpose(context_vec, 0, 1)
        return self.match_lstm(context_vec, context_lengths, question_vec, question_lengths)


class Embedding(torch.nn.Module):
    def __init__(self, device, word_embedding=False, part_of_speech=False, knowledge_graph=False):
        super().__init__()
        self.embedding = {}

        if isinstance(word_embedding, torch.Tensor):
            word_embedding.to(device)
            self.embedding['word'] = torch.nn.Embedding.from_pretrained(word_embedding)
            self.embedding['word'].to(device)
        if isinstance(part_of_speech, torch.Tensor):
            part_of_speech.to(device)
            self.embedding['pos'] = torch.nn.Embedding.from_pretrained(part_of_speech)
            self.embedding['pos'].to(device)
        if isinstance(knowledge_graph, torch.Tensor):
            knowledge_graph.to(device)
            self.embedding['kg'] = torch.nn.Embedding.from_pretrained(knowledge_graph)
            self.embedding['kg'].to(device)

    def forward(self, word_context, word_question, kg_context, kg_question, pos_context, pos_question):
        context = self.apply_embedding(word_context, pos_context, kg_context)
        question = self.apply_embedding(word_question, pos_question, kg_question)

        return torch.cat(context, dim=2), torch.cat(question, dim=2)

    def apply_embedding(self, word, pos, kg):
        applied_embeddings = []
        if 'word' in self.embedding:
            applied_embeddings.append(self.embedding['word'](word))
        if 'pos' in self.embedding:
            applied_embeddings.append(self.embedding['pos'](pos))
        if 'kg' in self.embedding:
            applied_embeddings.append(self.embedding['kg'](kg))

        return applied_embeddings
