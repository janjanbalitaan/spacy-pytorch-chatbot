from utils.trainers import trainer, get_xy_train, get_loader, train, save
from utils.utilities import remove_ignore_words, sort_word_list, sort_tags
from utils.constants import get_batch_size, get_hidden_size, get_learning_rate, get_num_epochs, get_ignore_words, get_device, get_is_shuffle, get_num_workers, get_criterion, get_optimizer
from datasets.chat import ChatDataset
from models.neuralnet import NeuralNet

import json

if __name__ == '__main__':
    filepath = 'data/client-2/data.json'
    savepath = 'data/client-2/data.pth'
    intents = []
    with open(filepath, 'r') as f:
        intents = json.load(f)

    tags, word_list, xy_data = trainer(intents['intents'])

    word_list = sort_word_list(remove_ignore_words(word_list))
    tags = sort_tags(tags)

    x_train, y_train = get_xy_train(xy_data, word_list, tags)

    output_size = len(tags)
    input_size = len(x_train[0])

    dataset = ChatDataset(x_train, y_train)
    train_loader = get_loader(dataset, get_batch_size(), get_is_shuffle(), get_num_workers())
    
    model = NeuralNet(input_size, get_hidden_size(), output_size).to(get_device())

    optimizer = get_optimizer(model.parameters(), get_learning_rate())
    
    if not train(get_num_epochs(), train_loader, get_device(), model, get_criterion(), optimizer):
        print('Error in training')
    
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": get_hidden_size(),
        "word_list": word_list,
        "tags": tags
    }

    if not save(savepath, data):
        print('Error in saving data')

    print('Training Complete')