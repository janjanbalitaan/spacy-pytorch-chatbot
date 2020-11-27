from models.neuralnet import NeuralNet
from utils.bag_of_words import bag_of_words
from utils.tokenizers import spacy_tokenizer
from utils.constants import get_device

import random
import json
import torch

if __name__ == "__main__":
    filepath = 'data/client-1/data.json'
    pthpath = 'data/client-1/data.pth'

    intents = []
    with open(filepath, 'r') as f:
        intents = json.load(f)

    data = torch.load(pthpath)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    word_list = data["word_list"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(get_device())
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "shop-bot"
    print("Let's chat! type 'quit' to exit")

    while True:
        text = input('You: ')

        if text == 'quit':
            break

        text = spacy_tokenizer(text)
        x = bag_of_words(text, word_list)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.7:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')

        else:
            print(f'{bot_name}: I do not understand...')


