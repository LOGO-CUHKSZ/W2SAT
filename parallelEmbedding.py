from multiprocessing import Pool
import os
import time
import random

import torch
import torch.nn as nn


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, literal_to_ix):
        super(CBOW, self).__init__()
        # out: 1 x embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.literal_to_ix = literal_to_ix
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_literal_embedding(self, literal):
        ix = torch.tensor([self.literal_to_ix[literal]])
        return self.embeddings(ix)

    def get_embeddings(self):
        ix = torch.tensor([i for i in range(self.vocab_size)])
        return self.embeddings(ix)


# utils
def make_context_vector(context, literal_to_idx):
    idxs = [literal_to_idx[l] for l in context]
    return torch.tensor(idxs, dtype=torch.long)


def read_sat(sat_path):
    with open(sat_path) as f:
        sat_lines = f.readlines()
        header = sat_lines[0]
        header_info = header.replace("\n", "").split(" ")
        num_vars = int(header_info[-2])
        num_clauses = int(header_info[-1])

        sat = [
            [int(x) for x in line.replace(" 0\n", "").split(" ")]
            for line in sat_lines[1:]
        ]

        return sat, num_vars, num_clauses


def getEmbedding(name):
    print(name)
    sat_path = f"./dataset/train_formulas/{name}"
    sat_instance, num_vars, num_clauses = read_sat(sat_path)
    vocab_size = num_vars * 2

    data = []
    for clause in sat_instance:
        clause_len = len(clause)
        for i in range(clause_len):
            context = [clause[x] for x in range(clause_len) if x != i]
            target = clause[i]
            data.append((context, target))

    print(f"data size: {len(data)}")

    # model setting
    EMDEDDING_DIM = 50

    literal_to_ix = {}
    for i in range(1, num_vars + 1):
        literal_to_ix[i] = 2 * i - 2
        literal_to_ix[-i] = 2 * i - 1

    model = CBOW(vocab_size, EMDEDDING_DIM, literal_to_ix)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # training
    for epoch in range(50):
        total_loss = 0
        for context, target in data:
            context_vector = make_context_vector(context, literal_to_ix)
            log_probs = model(context_vector)
            total_loss += loss_function(
                log_probs, torch.tensor([literal_to_ix[target]])
            )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, total_loss.item())

    # test the embedding
    embeddings = model.get_embeddings()
    torch.save(embeddings, f"./model/embeddings/{name}.pt")


if __name__ == "__main__":
    print("Parent process %s." % os.getpid())
    names = os.listdir("./dataset/train_formulas/")
    print(names)
    p = Pool(20)
    for name in names:
        p.apply_async(getEmbedding, args=(name,))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")
