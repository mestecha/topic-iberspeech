import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import logging
import warnings; warnings.simplefilter('ignore')

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import nltk

nltk.download('punkt')
logging.basicConfig(level=logging.INFO)
THRESHOLD = 0.80



##########################################
############### LOAD MODEL ###############
##########################################

# model_name = "cross-encoder/nli-deberta-base"
# model_name_save = 'deberta'
# model_name = "cross-encoder/nli-deberta-v3-base"
# model_name_save = 'deberta-v3'
model_name = "facebook/bart-large-mnli"
model_name_save = 'bart'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)



##########################################
############### RUN MODEL ################
##########################################

required_context = ['text']

def _parse_labels(labels):
    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split(",")]
    return labels

def _arg_parser(sequences, labels, hypothesis_template):
    if isinstance(sequences, str):
        sequences = [sequences]
    labels = _parse_labels(labels)

    sequence_pairs = []
    for sequence in sequences:
        sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

    return sequence_pairs

def _parse_and_tokenize(
        sequences,
        candidate_labels,
        hypothesis_template,
        padding=True,
        add_special_tokens=True,
        truncation="only_first"
):
    """
    Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
    """
    sequence_pairs = _arg_parser(sequences, candidate_labels, hypothesis_template)
    inputs = tokenizer(
        sequence_pairs,
        add_special_tokens=add_special_tokens,
        return_tensors='pt',
        padding=padding,
        truncation=truncation,
    )

    return inputs

def get_required_context():
    return required_context

def handle_message(msg, candidate_labels, hypothesis):
    """
    Topics: Detect the topic of a sentence
    """
    
    # print(f'-----------------------{msg}----------------')
    message = msg['text']

    inputs = _parse_and_tokenize(message,
                        candidate_labels,
                        hypothesis)
    with torch.no_grad():
        outputs = model(**inputs.to(device))[0].detach().cpu().clone().numpy()

    num_message = len(message)
    candidate_labels = _parse_labels(candidate_labels)
    reshaped_outputs = outputs.reshape((num_message, len(candidate_labels), -1))

    # softmax over the entailment vs. contradiction dim for each label independently
    entail_contr_logits = reshaped_outputs[..., [0, -1]]
    scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
    scores = scores[..., 1]

    result = []
    for iseq in range(num_message):
        top_inds = list(reversed(scores[iseq].argsort()))
        result.append(
            {
                "sequence": message if num_message == 1 else message[iseq],
                "labels": [candidate_labels[i] for i in top_inds],
                "scores": scores[iseq, top_inds].tolist(),
            }
        )

    if len(result) == 1:
        result = result[0]
    
    ans_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
    sorted_dict = dict(sorted(ans_dict.items(), key=lambda x: x[1], reverse=True))
    topic_dict = {k: v for i, (k, v) in enumerate(sorted_dict.items()) if i < 1}
    
    return list(topic_dict.keys())[0], list(topic_dict.values())[0]



##########################################
############### LOAD LABELS ##############
##########################################

candidate_labels_first_level = ['sports', 'politics', 'movies', 'books', 'music', 'science', 'animals', 'videogames']
candidate_labels_sports = ['football', 'basketball', 'tennis', 'player', 'coach']
candidate_labels_politics = ['trump', 'elections', 'vote', 'biden', 'poll', 'fake news']
candidate_labels_movies = ['genre', 'director', 'title', 'plot', 'actor', 'synopsis']
candidate_labels_books = ['genre', 'author', 'title', 'plot', 'harry potter']
candidate_labels_music = ['genre','song','singer', 'band', 'lyrics', 'rythm', 'dance']
candidate_labels_science = ['nature', 'maths', 'computer', 'physics', 'space', 'robots']
candidate_labels_animals = ['animals']
candidate_labels_videogames = ['arcade', 'console', 'play station', 'xbox', 'wii']
def get_subtopic_first_level(topic):
    if topic == 'sports':
        candidate_labels_subtopic = candidate_labels_sports
    elif topic == 'politics':
        candidate_labels_subtopic = candidate_labels_politics
    elif topic == 'movies':
        candidate_labels_subtopic = candidate_labels_movies
    elif topic == 'books':
        candidate_labels_subtopic = candidate_labels_books
    elif topic == 'music':
        candidate_labels_subtopic = candidate_labels_music
    elif topic == 'science':
        candidate_labels_subtopic = candidate_labels_science
    elif topic == 'videogames':
        candidate_labels_subtopic = candidate_labels_videogames
    else:
        candidate_labels_subtopic = ''
    return candidate_labels_subtopic

candidate_labels_second_level = ['art', 'teacher', 'family', 'finance', 'cars', 'astronomy']
candidate_labels_art = ['ballet', 'theatre', 'cinema', 'museum', 'painting']
candidate_labels_teacher = ['history', 'school', 'subject', 'university', 'mark','professor']
candidate_labels_family = ['parents', 'friends', 'relatives', 'marriage', 'sons']
candidate_labels_finance = ['bitcoins', 'investment', 'stock market', 'benefits', 'finances']
candidate_labels_cars = ['electric vehicle', 'fuel', 'speed', 'model']
candidate_labels_astronomy = ['astronomy']
def get_subtopic_second_level(topic):
    if topic == 'art':
        candidate_labels_subtopic = candidate_labels_art
    elif topic == 'teacher':
        candidate_labels_subtopic = candidate_labels_teacher
    elif topic == 'family':
        candidate_labels_subtopic = candidate_labels_family
    elif topic == 'finance':
        candidate_labels_subtopic = candidate_labels_finance
    elif topic == 'cars':
        candidate_labels_subtopic = candidate_labels_cars
    elif topic == 'astronomy':
        candidate_labels_subtopic = candidate_labels_astronomy
    else:
        candidate_labels_subtopic = ''
    return candidate_labels_subtopic

candidate_labels_third_level = ['food', 'photography', 'newspaper', 'fashion', 'climate conditions', 'facebook']
candidate_labels_food = ['healthy', 'vegetables', 'fish', 'meat', 'dessert']
candidate_labels_photography = ['camera', 'light', 'lens', 'zoom', 'optics']
candidate_labels_newspaper = ['press', 'interview', 'exclusive', 'trending', 'fake news']
candidate_labels_fashion = ['model', 'clothes', 'dress', 'jewel', 'catwalk', 'design']
candidate_labels_climate_conditions = ['sunny', 'cloudy', 'raining', 'cold', 'hot']
candidate_labels_facebook = ['twitter', 'instagram', 'facebook', 'fake news']
def get_subtopic_third_level(topic):
    if topic == 'food':
        candidate_labels_subtopic = candidate_labels_food
    elif topic == 'photography':
        candidate_labels_subtopic = candidate_labels_photography
    elif topic == 'newspaper':
        candidate_labels_subtopic = candidate_labels_newspaper
    elif topic == 'fashion':
        candidate_labels_subtopic = candidate_labels_fashion
    elif topic == 'climate conditions':
        candidate_labels_subtopic = candidate_labels_climate_conditions
    elif topic == 'facebook':
        candidate_labels_subtopic = candidate_labels_facebook
    else:
        candidate_labels_subtopic = ''
    return candidate_labels_subtopic

candidate_labels_amazon = ['Sports', 'Politics', 'Entertainment_Movies', 'Entertainment_Books', 'Entertainment_Music', 'Science_and_Technology', 'Entertainment_General', 'Phatic', 'Interactive', 'Inappropriate_Content', 'Other']
candidate_labels_amazon_uncased = ['sports', 'politics', 'movies', 'books', 'music', 'science', 'general', 'phatic', 'interactive', 'inappropriate', 'other']



##########################################
################ LOAD DATA ###############
##########################################

df_amazon_turn = pd.read_csv('data/df_amazon_turn.csv')



##########################################
################## MAIN ##################
##########################################

start = time.time()
turn_topic_1, turn_topic_score_1 = [], []
turn_topic_2, turn_topic_score_2 = [], []
turn_topic_3, turn_topic_score_3 = [], []
turn_subtopic_1, turn_subtopic_score_1 = [], []
turn_subtopic_2, turn_subtopic_score_2 = [], []
turn_subtopic_3, turn_subtopic_score_3 = [], []
turn_topic_amazon, turn_topic_amazon_score  = [], []
turn_topic_amazon_uncased, turn_topic_amazon_uncased_score  = [], []
hypothesis = 'The topic of this sentence is {}.'

for turn in tqdm(df_amazon_turn['USER']):
    utterance = {'text': [turn]}

    # First level
    CANDIDATE_LABELS = candidate_labels_first_level
    topic, topic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    turn_topic_1.append(topic)
    turn_topic_score_1.append(topic_score)

    candidate_labels_subtopic = get_subtopic_first_level(topic)
    if candidate_labels_subtopic:
        CANDIDATE_LABELS = candidate_labels_subtopic
        subtopic, subtopic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    else:
        subtopic, subtopic_score = topic, 0
    turn_subtopic_1.append(subtopic)
    turn_subtopic_score_1.append(subtopic_score)

    # Second level
    CANDIDATE_LABELS = candidate_labels_second_level
    topic, topic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    turn_topic_2.append(topic)
    turn_topic_score_2.append(topic_score)

    candidate_labels_subtopic = get_subtopic_second_level(topic)
    if candidate_labels_subtopic:
        CANDIDATE_LABELS = candidate_labels_subtopic
        subtopic, subtopic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    else:
        subtopic, subtopic_score = topic, 0
    turn_subtopic_2.append(subtopic)
    turn_subtopic_score_2.append(subtopic_score)

    # Third level
    CANDIDATE_LABELS = candidate_labels_third_level
    topic, topic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    turn_topic_3.append(topic)
    turn_topic_score_3.append(topic_score)

    candidate_labels_subtopic = get_subtopic_third_level(topic)
    if candidate_labels_subtopic:
        CANDIDATE_LABELS = candidate_labels_subtopic
        subtopic, subtopic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    else:
        subtopic, subtopic_score = topic, 0
    turn_subtopic_3.append(subtopic)
    turn_subtopic_score_3.append(subtopic_score)

    # Amazon
    CANDIDATE_LABELS = candidate_labels_amazon
    topic, topic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    turn_topic_amazon.append(topic)
    turn_topic_amazon_score.append(topic_score)

    # Amazon uncased
    CANDIDATE_LABELS = candidate_labels_amazon_uncased
    topic, topic_score = handle_message(utterance, CANDIDATE_LABELS, hypothesis)
    turn_topic_amazon_uncased.append(topic)
    turn_topic_amazon_uncased_score.append(topic_score)

df_amazon_turn['TOPIC_IS_1'] = turn_topic_1
df_amazon_turn['TOPIC_IS_SCORE_1'] = turn_topic_score_1
df_amazon_turn['SUBTOPIC_IS_1'] = turn_subtopic_1
df_amazon_turn['SUBTOPIC_IS_SCORE_1'] = turn_subtopic_score_1
df_amazon_turn['TOPIC_IS_2'] = turn_topic_2
df_amazon_turn['TOPIC_IS_SCORE_2'] = turn_topic_score_2
df_amazon_turn['SUBTOPIC_IS_2'] = turn_subtopic_2
df_amazon_turn['SUBTOPIC_IS_SCORE_2'] = turn_subtopic_score_2
df_amazon_turn['TOPIC_IS_3'] = turn_topic_3
df_amazon_turn['TOPIC_IS_SCORE_3'] = turn_topic_score_3
df_amazon_turn['SUBTOPIC_IS_3'] = turn_subtopic_3
df_amazon_turn['SUBTOPIC_IS_SCORE_3'] = turn_subtopic_score_3
df_amazon_turn['TOPIC_AMAZON_IS'] = turn_topic_amazon
df_amazon_turn['TOPIC_AMAZON_IS_SCORE'] = turn_topic_amazon_score
df_amazon_turn['TOPIC_AMAZON_UNCASED_IS'] = turn_topic_amazon_uncased
df_amazon_turn['TOPIC_AMAZON_UNCASED_IS_SCORE'] = turn_topic_amazon_uncased_score

end = time.time()
print(end-start)

logging.info('Saving Results')
df_amazon_turn.to_csv('results/df_amazon_turn_' + model_name_save + '_topics.csv', index=False)
