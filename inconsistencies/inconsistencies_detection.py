import logging
import sys
import random
import numpy as np
import umap
import hdbscan
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import plotly.express as px

logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

DATA_PATH = ''

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
zsl_model = pipeline("zero-shot-classification", model = "cross-encoder/nli-deberta-base")

def get_most_frequent_question_embedding(questions_df, embedding_model):
    dict_result = {
          'main_question':questions_df.turn1.mode()[0],
          'main_embedding':embedding_model.encode(questions_df['turn1'].mode()[0])
    }
    print(f'Most frequent question: {questions_df.turn1.mode()[0]}')
    return dict_result

def computing_embedding_list(questions):
    list_embeddings = []
    for sent in questions:
        list_embeddings.append(embedding_model.encode(sent))
    return list_embeddings

def filter_similar_questions(questions, embedding_model, main_question, main_embedding, threshold=0.9):
    dict_results = {
        'list_chatbot':[],
        'list_embeddings':[],
        'list_questions':[],
        'list_answers':[],
        'similarity':[]
    }
    aux = questions[questions['turn1'] != main_question]
    aux.reset_index(inplace=True, drop=True)
    for i in range(aux.shape[0]):
        aux_emb = embedding_model.encode(aux['turn1'][i])
        if cosine_similarity(aux_emb.reshape(1,-1), main_embedding.reshape(1,-1)) > threshold:
            dict_results['list_chatbot'].append(aux['selected'][i])
            dict_results['list_embeddings'].append(aux_emb)
            dict_results['list_questions'].append(aux['turn1'][i])
            dict_results['list_answers'].append(aux['turn2'][i])
            dict_results['similarity'].append(cosine_similarity(aux_emb.reshape(1,-1), main_embedding.reshape(1,-1)))
            # print(f'{aux.turn1[i]}. Similarity: {cosine_similarity(aux_emb.reshape(1,-1), main_embedding.reshape(1,-1))}')
    return dict_results

def extracting_answers(questions_df, filter_dict_result, main_question):
    list_answers = []
    aux = questions_df[questions_df['turn1'] == questions_df['turn1'].mode()[0]]['turn2']
    list_answers.extend(aux)
    list_answers.extend(filter_dict_result['list_answers'])
    return list_answers

def extracting_answers_and_chatbot(questions_df, filter_dict_result, main_question):
    dict_answers_chatbot = {
        'list_answers':[],
        'list_chatbot':[]
    }
    aux = questions_df[questions_df['turn1'] == questions_df['turn1'].mode()[0]]['turn2']
    dict_answers_chatbot['list_answers'].extend(aux)
    dict_answers_chatbot['list_answers'].extend(filter_dict_result['list_answers'])
    aux = questions_df[questions_df['turn1'] == questions_df['turn1'].mode()[0]]['selected']
    dict_answers_chatbot['list_chatbot'].extend(aux)
    dict_answers_chatbot['list_chatbot'].extend(filter_dict_result['list_chatbot'])
    return dict_answers_chatbot

def computing_embedding_list(list_):
    list_embeddings = []
    for sent in list_:
        list_embeddings.append(embedding_model.encode(sent))
    return list_embeddings

def mean_embedding(list_embeddings):
    return np.mean(np.array(list_embeddings),axis=0)

def reduce_dimensionality(list_embeddings, mean_embedding_vector):
    list_emb = list_embeddings
    list_emb.append(mean_embedding_vector)
    reducer = umap.UMAP(random_state=42,n_components=2)
    try: 
        list_emb_reduce = reducer.fit_transform(list_emb)
    except:
        list_emb_reduce = list_emb
    return list_emb_reduce

def reduce_dimensionality_by_chatbot(dict_list, list_embedding_answers, mean_embedding_vector):
    dict_list['list_embeddings'] = list_embedding_answers
    df_ = pd.DataFrame(dict_list)
    list_df = []
    for chatbot in df_['list_chatbot'].unique():
        reducer = umap.UMAP(random_state=42,n_components=2)
        aux_df = df_[df_['list_chatbot']==chatbot]
        aux_df.reset_index(inplace=True, drop=True)
        list_emb = list(aux_df['list_embeddings'].values)
        list_chatbot = [chatbot]*len(list_emb)
        list_chatbot.append('MEAN_VECTOR')
        list_emb.append(mean_embedding_vector)
        list_emb_reduce = reducer.fit_transform(list_emb)
        list_answers_ = list(aux_df['list_answers'].values)
        list_answers_.append('MEAN VECTOR')
        chatbot_df = pd.DataFrame(list(zip(list_emb_reduce[:,0], list_emb_reduce[:,1], 
                                           list_chatbot,list_answers_)),
                                  columns=['comp1','comp2','chatbot', 'list_answers'])
        list_df.append(chatbot_df)
    return list_df

def scale_data(list_embeddings):
    scaler = StandardScaler()
    return scaler.fit_transform(list_embeddings)

def cluster_metrics(embeddings_vectors):
    cluster = hdbscan.HDBSCAN(min_cluster_size=int(len(embeddings_vectors)/2),
                              metric='euclidean', # Cosine distance
                              allow_single_cluster=True,                    
                              cluster_selection_method='eom').fit(embeddings_vectors)
    sil = silhouette_score(embeddings_vectors, cluster.labels_)
    print(f'Number of clusters: {max(cluster.labels_+1)}')
    dict_results = {
        'probabilities':cluster.probabilities_,
        'persistance':cluster.cluster_persistence_,
        'silhouette_score': sil
    }
    return dict_results

def cluster_representation(list_answers, list_emb_reduce):
    df_representation = pd.DataFrame()

    is_mean_vector = len(list_answers)*[0]
    is_mean_vector.append(1)

    aux = list_answers
    aux.append('MEAN EMBEDDING')

    df_representation['comp1'] = list_emb_reduce[:,0]
    df_representation['comp2'] = list_emb_reduce[:,1]
    df_representation['answers'] = aux
    df_representation['is_mean_vector'] = is_mean_vector
    
    fig = px.scatter(df_representation, x="comp1", y="comp2",
                 color="is_mean_vector",
                 hover_data=["answers"])
    
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_layout(
            title="What's your favorite movie?",
            xaxis_title="X Dimension",
            yaxis_title="Y Dimension",
            font=dict(
                size=18,
            )
        )
    fig.show()
    return 0

def get_silhouette_score(list_emb_reduce_scale):
    sil = []
    kmax = 10
    if len(list_emb_reduce_scale) < kmax+1:
        kmax=len(list_emb_reduce_scale)-1
    representation = None
    if kmax >1:
        representation = True
        for k in range(2,kmax+1):
            kmeans = KMeans(n_clusters=k).fit(list_emb_reduce_scale)
            labels = kmeans.labels_
            sil.append(silhouette_score(list_emb_reduce_scale, labels, metric='euclidean'))
    else:
        sil = list(np.arange(0,kmax))
    
    if representation == True:
        fig = px.line(x=np.arange(2,kmax+1), y=sil,  markers=True)

        fig.update(layout_yaxis_range = [0,1])
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        fig.update_layout(
            title="What's your favorite movie?",
            xaxis_title="Number of groups",
            yaxis_title="Silhouette score",
            font=dict(
                size=18,
            )
        )
        fig.show()
    return sil

def get_n_clusters(list_silhouette):
    print(f'There are {list_silhouette.index(max(list_silhouette))+2} possible answers')
    return list_silhouette.index(max(list_silhouette))+2

def compute_median(list_embedding_scaled, indexes):
    vectors = []
    for i in indexes:
        vectors.append(list_embedding_scaled[i])
    return vectors, np.median(vectors, axis=0)

def compute_mean(list_embedding_scaled, indexes):
    vectors = []
    for i in indexes:
        vectors.append(list_embedding_scaled[i])
    return vectors, np.mean(vectors, axis=0)

def get_main_index_vector(vectors, median_vector):
    vectors = np.asarray(vectors)
    dist_2 = np.sum((vectors - median_vector)**2, axis=1)
    return np.argmin(dist_2)

def get_main_answer(n_clusters, list_emb_reduce,
                    list_answers, metric='median'):
    if metric != 'median' and metric != 'mean':
        sys.exit('Metric must be median or mean')
    if metric=='median':
        function_to_apply = compute_median
    else:
        function_to_apply = compute_mean

    kmeans = KMeans(n_clusters=int(n_clusters))
    kmeans.fit(list_emb_reduce)
    labels = kmeans.labels_
    list_emb_reduce_np = list_emb_reduce[:-1]
    main_vectors = []
    main_answers=[]
    for label in np.unique(labels):
        indexes = [i for i, value in enumerate(labels[:-1]) if value == label]
        vectors, median = function_to_apply(list_emb_reduce_np, indexes)
        answers = [list_answers[i] for i in indexes]
        main_vectors.append(vectors[get_main_index_vector(vectors, median)])
        main_answers.append(answers[get_main_index_vector(vectors, median)])
    return main_vectors, main_answers

def get_chatbot_likes(df_user_questions, topic_target, embedding_model,
                      threshold=0.95, metric='median', min_answers=1, 
                      representation=True):
    questions_topic = df_user_questions[df_user_questions['bertopic'] == topic_target]
    questions_topic.reset_index(inplace=True, drop=True)
    
    most_frequent_dict = get_most_frequent_question_embedding(questions_topic, embedding_model)
    filter_similar_dict = filter_similar_questions(questions_topic,embedding_model, 
                                               most_frequent_dict['main_question'],
                                               most_frequent_dict['main_embedding'], 
                                               threshold)
    list_answers = extracting_answers(questions_topic, filter_similar_dict, 
                                      most_frequent_dict['main_question'])
    
    if len(list_answers) <= 1:
        print(f'TOPIC {topic_target}: THERE IS NOT QUESTIONS')
        return list_answers

    if len(list_answers) <= min_answers:
        aux_answer = max(set(list_answers), key = list_answers.count)
        if list_answers.count(aux_answer) == 1:
            return random.choice(list_answers)
        else: 
            return aux_answer

    dict_answers_and_chatbots = extracting_answers_and_chatbot(questions_topic, 
                                                           filter_similar_dict, 
                                                           most_frequent_dict['main_question'])
    list_embedding_answers = computing_embedding_list(dict_answers_and_chatbots['list_answers'])
    dict_answers_and_chatbots['list_embeddings'] = list_embedding_answers
    mean_embedding_answer = mean_embedding(list_embedding_answers)
    list_emb_reduce = reduce_dimensionality(list_embedding_answers, mean_embedding_answer)
    list_silhouette_score = get_silhouette_score(list_emb_reduce)
    n_clusters = get_n_clusters(list_silhouette_score)
    scaler = StandardScaler()
    list_emb_reduce_scale = scaler.fit_transform(list_emb_reduce)
    main_chatbot_answer_vectors, main_answers = get_main_answer(n_clusters, list_emb_reduce_scale, 
                                                            dict_answers_and_chatbots['list_answers'], 
                                                            metric=metric)
    if representation == True:
        cluster_representation(list_answers, list_emb_reduce_scale)
    return main_answers


def get_same_topic_questions(zsl_model, df_user_questions, topic):
    candidates = []
    candidates.append(topic)
    candidates.append('other')
    zs_topic = []
    zs_score = []
    for question in df_user_questions['turn1']:
        model_output = zsl_model(question, candidates, multi_label=True)
        if model_output['labels'][0] == topic:
            zs_topic.append(model_output['labels'][0])
            zs_score.append(model_output['scores'][0])
        else:
            zs_topic.append(model_output['labels'][1])
            zs_score.append(model_output['scores'][1])
    df_user_questions['zs_topic'] = zs_topic
    df_user_questions['zs_score'] = zs_score
    return df_user_questions

def filter_zs_topic(df_user_questions, zs_threshold):
    resulting_df = df_user_questions[df_user_questions['zs_score']>zs_threshold]
    resulting_df.reset_index(inplace=True, drop=True)
    return resulting_df

def get_chatbot_topic_likes(df_user, topic_like:str, topic_model,
                            embedding_model, question_type='favorite',
                            threshold=0.95, metric = 'median', 
                            min_answers=1, representation=False,
                            zs_threshold=0.7):
    similar_topic, similarity = topic_model.find_topics(topic_like, top_n=10)
    if question_type == 'like':
        topic_like_question = 'do you like '+topic_like+'?'
    elif question_type=='favorite':
        topic_like_question = "what's your favorite "+topic_like+'?'
    elif question_type == 'no-fill':
        topic_like_question = topic_like
    else:
        sys.exit('Question type must be like or favorite')
    question_embedding = embedding_model.encode(topic_like_question)
    for topic in similar_topic:
        for representative_doc in topic_model.get_representative_docs(topic):
            aux = embedding_model.encode(representative_doc)
            cos_sim = cosine_similarity(aux.reshape(1,-1), 
                                        question_embedding.reshape(1,-1))
            if cos_sim >= 0.65:
                print(f'Most representative doc: {topic_model.get_representative_docs(topic)}')
                print(f'Cosine similarity: {cos_sim}')
                print(f"Performing topic filtering by zero-shot")

                df_user = get_same_topic_questions(zsl_model, df_user, topic)
                df_user = filter_zs_topic(df_user, zs_threshold)

                if df_user.shape[0] > 0:
                    main_answers = get_chatbot_likes(df_user, topic,
                                                 embedding_model,
                                                 threshold=threshold,
                                                 metric=metric,
                                                 min_answers=min_answers,
                                                 representation=representation)
                    for ans in main_answers:
                        print(f'- {ans}')
                    break
    return

if __name__ == '__main__':
    logging.info('Loading conversations')
    data = pd.read_csv(DATA_PATH, index_col=0)
    data_user_questions = data[data['turn1'].str.contains("[?]")]
    data_user_questions.reset_index(inplace=True, drop=True)
    logging.info('Loading BERTopic model')
    topic_model = BERTopic()
    topic_model = BERTopic.load()
