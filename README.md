# About
The project aims to provide multi-lingual chatbot using spacy and pytorch. The bot was created by following the tutorial of [@python-engineer](https://github.com/python-engineer/pytorch-chatbot).

# Technology Requirements
- [Python 3.8.5](https://www.python.org/downloads/release/python-385/)
- [Virtual Environment](https://virtualenv.pypa.io/en/latest/)

# Dependencies
- [spacy](https://spacy.io/) 2.3 or higher

# Installation
- Python Dependencies
    - virtualenv -p python3.8.5 venv
    - source venv/bin/activate
    - pip install -r reuquirements.txt
- Spacy Dependencies
    - To install chinese language support
        - python -m spacy download zh_core_web_sm
    - To install english language support
        -  python -m spacy download en_core_web_sm
    - To install multi-language support
        - python -m spacy download xx_ent_wiki_sm


# DataSets
- Main dataset
``` ts
    type DataSet = {
        intents: array:Intent
    }
```
- Intent dataset
``` ts
    type Intent = {
        tag: string,
        patterns: array:string,
        responses: array:string
    }
```

# How to run
- Pre-requisites
    - Training datasets following the Datasets format given above
    - Follow the installtion guide

- Run
    - Train first the model
        - python train.py
    - Run the chat function to make sure everything is okay
        - python app.py


