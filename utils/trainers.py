from .tokenizers import spacy_tokenizer
from .bag_of_words import bag_of_words

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def trainer(intents):
    word_list = []
    tags = []
    xy_data = []

    for intent in intents:
        tag = intent['tag']
        tags.append(tag)

        for pattern in intent['patterns']:
            word = spacy_tokenizer(pattern)
            word_list.extend(word)
            xy_data.append((word, tag))

    return tags, word_list, xy_data


def get_xy_train(xy_data, word_list, tags):
    x_train = []
    y_train = []
    for (pattern_sentence, tag) in xy_data:
        bog = bag_of_words(pattern_sentence, word_list)

        x_train.append(bog)
        label = tags.index(tag)
        y_train.append(label)

    return np.array(x_train), np.array(y_train)

def get_loader(dataset, batch_size, is_shuffle, num_workers):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)

def train(num_epochs, train_loader, device, model, criterion, optimizer):
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch+1}/num_epochs, loss={loss.item():.4f}')

    return True

def save(path, data):
    torch.save(data, path)
    return True