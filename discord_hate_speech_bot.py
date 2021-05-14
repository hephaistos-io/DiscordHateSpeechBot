import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import argparse
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import regex as re


import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

bot = commands.Bot(command_prefix='?', description="This is a hate speech classification bot")

STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words('english')
MENTION_REGEX = "<@[0-9]+>"
MENTION_FINDER = re.compile(MENTION_REGEX)

def parse_text(text):
    text = [w for w in text if not w in STOPWORDS]
    text = [w for w in text if not re.sub('\'\.,','',w).isdigit()]
    text = [STEMMER.stem(w) for w in text]
    text = ' '.join(text)
    return text

def generate_hate_score_embed(message, hateScore):
    embed=discord.Embed(title="Hate Speech Classifier", description="Checks the hate speech score of your supplied message. This is mostly used for verification", color=0xc62424)
    embed.set_author(name=message.author.name, icon_url=message.author.avatar_url)
    embed.add_field(name="Input Text", value=message.content, inline=False)
    embed.add_field(name="Hate Speech Score", value=str(hateScore[0][0]), inline=False)
    embed.add_field(name="Classified as Hate? (score >= 0)", value=str(hateScore[0][0] >= 0), inline=False)
    embed.set_footer(text="Think the score is wrong? Contact the creator")
    return embed

def generate_help_embed():
    embed=discord.Embed(title="Hate Speech Classifier", description="A bot that checks your messages for hate speech", color=0xc62424)
    embed.set_author(name=bot.user.name, icon_url=bot.user.avatar_url)
    embed.add_field(name="Overview", 
        value="The bot checks every new message and runs it trough a trained model, that calculates it's hate score.", 
        inline=False)
    embed.add_field(name="Classification", 
        value="The classification is simple. The models calculated score is checked agains `>=0`, if it passes, it's hate speech", 
        inline=False)
    embed.add_field(name="Wrong classification", 
        value="As with everything, wrong classifications can't be avoided. There are features being planned, where you can " +
        "execute a function to give direct feedback - this would then add the message + its classification to the training pool", 
        inline=False)
    embed.add_field(name="Model", 
        value="The model used was built using TensorFlow and is heavily based on this tutorial: https://www.tensorflow.org/tutorials/text/text_classification_rnn", 
        inline=False)
    embed.add_field(name="Training data", 
        value="The training data used is a collection of Wikipedia comments that are classified. We reduced the classifications from 7 to 2: "+
        "Hate Speech and non-Hate Speech. The data can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data", 
        inline=False)
    embed.add_field(name="How good is the model?",
    value="""This are the current scores of the latest training:\n
    ```
                     precision    recall  f1-score   support

non-hate speech       0.96      0.99      0.97      5734
    hate-speech       0.84      0.64      0.73       649

       accuracy                           0.95      6383
      macro avg       0.90      0.81      0.85      6383
   weighted avg       0.95      0.95      0.95      6383

Using Matthews correlation coefficient to measure quality
MCC-Quality:  0.7091936368873746    
    ```""")
    embed.add_field(name="Functions", 
        value="Use `?about` to get this message again.\n Mention this Bot in any message and it will create and embed with its classification.",
        inline=False)
    return embed

@bot.command()
async def about(ctx):
    await ctx.reply(embed=generate_help_embed())

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.listen()
async def on_message(message):
    if message.author == bot.user:
        return
    
    text = re.sub(MENTION_FINDER, '', message.content)
    parsed_text = parse_text(text)
    print(parsed_text)
    hateScore = model.predict([parsed_text])
    if bot.user in message.mentions:
        await message.reply(embed = generate_hate_score_embed(message, hateScore))
    else:
        if hateScore >= 0:
            await message.reply("Your post has been classified as hate speech")
            

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train tensorflow-network on hate-speech")
    parser.add_argument('-l', '--load', help="loads tensorflow model", required=True)
    return parser.parse_args(argv)

def main(argv):

    print('DiscordHateSpeechBot')
    global model
    args = parse_args(argv)
    #make sure it doesnt run on GPU
    tf.config.set_visible_devices([], 'GPU')

    print('Loading existing model')
    model = tf.keras.models.load_model(args.load)
    print('Finished loading model, starting client')
    bot.run(TOKEN)

if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))

            
