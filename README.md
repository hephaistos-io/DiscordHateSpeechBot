### DiscordHateSpeechBot
This is a TF Model that is trained to classify hate speech. It contains a script to run a bot on discord, that will automatically check posts for their hate-score

### HowTo

# Installation

Make sure you have every needed library installed. You need:
- Python
- pip

With pip you can install:

- matplotlib
- discord py
- tensorflow
- sklearn
- numpy

# Training your first model

For that you want to use `tensorflow_model_creation.py`. These are the arguments:
```
usage: tensorflow_model_creation.py [-h] [-ds DOWNSAMPLE] [-d] [-c]
                                    [-e EPOCHS] [-r] [-rs RANDOMSTATE]
                                    [-l LOAD] [-ct]

Train tensorflow-network on hate-speech

optional arguments:
  -h, --help            show this help message and exit
  -ds DOWNSAMPLE, --downsample DOWNSAMPLE
                        downsample data, has to be a float between (0, 1) (exclusive)
  -d, --debug           turns debug on
  -c, --cpu             sets runmode to cpu only
  -e EPOCHS, --epochs EPOCHS
                        sets how many epochs the model shall train
  -r, --reprocess       activates the reprocessing of the training data. Has to be run the first time!
  -rs RANDOMSTATE, --randomstate RANDOMSTATE
                        sets a specific random state, used for the random.seed
  -l LOAD, --load LOAD  loads tensorflow model under the given path
  -ct, --continuetraining
                        If a model to load has been passed, this parameter
                        will continue training it. It is advised to supply a
                        new -rs value, to mix up the training data
```

The script then does everything by itself, no need for you to intervene. The first launch should look like this:
`python tensorflow_model_creation.py -r`


# Running the discord bot

First, you need to turn the `.env_template` file into a `.env` file and add your bots token.
Then, you can run the script with: `python discord_hate_speech_bot.py -l pathToYourSavedModel`

***IMPORTANT:***

If you are training your own model, make sure to copy the resulting validation scores from the CLI and paste them into the `about` section of the code. If you don't change/remove it, the score will be wrong. 

<hr>

## General information

The model is a `TensorFlow` RNN and is heavily based on this one:
https://www.tensorflow.org/tutorials/text/text_classification_rnn

The dataset can be found here: 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The dataset for training is being reduced from 6 labels to only 1.

<hr>

## ToDo:

- Make model save a `latest_model` that can be automatically loaded into the bot
- Make model save validation data so the bot can load it in automatically
- Create function for users to give direct feedback that allows the addition of new labeled datasets