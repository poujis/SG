from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import keras.losses as losses
import keras.metrics as metrics

from gensim.models import Word2Vec

import re
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import time
import timeit

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data
#
train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')
#
# print(len(train_data))
# print(len(train_data[0]))
# print(len(test_data))
# print(len(test_data[0]))
#
# # 1) morphs : 형태소 추출
# # 2) pos : 품사 부착(Part-of-speech tagging)
# # 3) nouns : 명사 추출
okt = Okt()
print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', encoding='UTF8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', encoding='UTF8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")


tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

# # 예쁘게(?) 출력하기 위해서 pprint 라이브러리 사용
# pprint(train_docs[0])
#

# print(len(tokens))
#

#
# # 전체 토큰의 개수
# print(len(text.tokens))
#
# # 중복을 제외한 토큰의 개수
# print(len(set(text.tokens)))
#
# # 출현 빈도가 높은 상위 토큰 10개
# pprint(text.vocab().most_common(10))

# 시간이 꽤 걸립니다! 시간을 절약하고 싶으면 most_common의 매개변수를 줄여보세요.
# most_common(100) 의 수를 높일 수록 정확도가 올라갑니다.

selected_words = [f[0] for f in text.vocab().most_common(5000)]


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

# #
# #
#
# train_x = [term_frequency(d) for d, _ in train_docs]
# test_x = [term_frequency(d) for d, _ in test_docs]
# train_y = [c for _, c in train_docs]
# test_y = [c for _, c in test_docs]
#
# x_train = np.asarray(train_x).astype('float32')
# x_test = np.asarray(test_x).astype('float32')
# y_train = np.asarray(train_y).astype('float32')
# y_test = np.asarray(test_y).astype('float32')
#
#
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(5000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
#
# model.fit(x_train, y_train, epochs=200, batch_size=512)
#
# results = model.evaluate(x_test, y_test)
#
# model.save('movie_eval.h5')
#
#
#
# ##########################################
# # 모델 아키텍처 따로 저장
# # 모델 아키텍처를 json 형식으로 저장
# model_json = model.to_json()
# # json 파일에서 모델 아키텍처 재구성
# with open("model.json", "w") as json_file :
#     json_file.write(model_json)
# # Weights 따로 저장
# model.save_weights("movie_eval.h5")
# print("Saved model to disk")
# # ##########################################

#######################################
# 위에서 저장했던 모델 아키텍처를 불러옵니다.
from keras.models import model_from_json

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("movie_eval.h5")
print("Loaded model from disk")
#############################################

#로드한 모델을 원래 형태로 컴파일 시켜줍니다
loaded_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                     loss=losses.binary_crossentropy,
                     metrics=[metrics.binary_accuracy])



def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(loaded_model.predict(data))  #loaded_model로 변경됨
    # if(score >= 0.75):
    #     text = "[{}]는 {:.2f}% 기쁨!!!!!!!!!.^^\n".format(review, score * 100)
    # elif (score >= 0.5):
    #     text = "[{}]는 {:.2f}% 좋음^^^^^^^^^^^^^^^\n".format(review, score * 100)
    # elif (score >= 0.25):
    #     text = "[{}]는 {:.2f}% 우울.ㅡㅡㅡㅡㅡㅡㅡ\n".format(review, score * 100)
    # else:
    #     text = "[{}]는 {:.2f}% 슬픔 ㅠㅠㅠㅠㅠㅠㅠ\n".format(review, (1 - score) * 100)
    if(score >= 0.75):
        text = "인천상륙작전을 추천합니다."
    elif (score >= 0.5):
        text = "뽀로로를 추천합니다."
    elif (score >= 0.25):
        text = "스펀지밥을 추천합니다."
    else:
        text = "쯔양의 먹방을 추천합니다."

    return text


# 테스트
predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("심심해")
predict_pos_neg("기뻐!! 신나!!")
predict_pos_neg("슬퍼 죽겠다")
predict_pos_neg("우울해")
# http://blog.naver.com/PostView.nhn?blogId=2feelus&logNo=220384206922&redirect=Dlog&widgetTypeCall=true
# [출처] 한글을 이용한 데이터마이닝및 word2vec이용한 유사도 분석|작성자 IDEO (참고하여 소스 커스터마이징)
# 1. 읽기
#!/usr/bin/env python
# -- coding: utf-8 --
from konlpy.corpus import kobill    # Docs from pokr.kr/bill
files_ko = kobill.fileids()         # Get file ids

# news.txt는 http://boilerpipe-web.appspot.com/ 를 통해 포탈뉴스 부분에서 긁어왔다.
# news.txt 는  konlpy의 corpus아래에 있는 kobill directory에 미리 저장되어있어야 한다.
# /Library/Python/2.7/site-packages/konlpy/data/corpus/kobill
doc_ko = kobill.open('ratings_train.txt').read()

# 2.Tokenize (의미단어 검출)
from konlpy.tag import Okt
import os
import json

# 학습시간이 오래 걸리므로 파일로 저장하여 처리 한다.
if os.path.isfile('tokens_ko_morphs.txt'):
    with open('tokens_ko_morphs.txt', encoding='UTF8') as f:
        tokens_ko_morphs = json.load(f)
else:
    okt = Okt()
    tokens_ko_morphs = okt.morphs(doc_ko)

    # JSON 파일로 저장
    with open('tokens_ko_morphs.txt', 'w', encoding="utf-8") as make_file:
        json.dump(tokens_ko_morphs, make_file, ensure_ascii=False, indent="\t")

# 3. Token Wapper 클래스 만들기(token에대해 이런 저런 처리를 하기 위해)
import nltk

# 일반적인 영화 평론의 기준과 유사한 단어를 추출한다.
# 스토리, 영상, 연출, 연기
# ko = nltk.Text(tokens_ko_morphs, name="연기")   # 이름은 아무거나
ko = nltk.Text(tokens_ko_morphs)

# 4. 토근 정보및 단일 토큰 정보 알아내기
print(len(ko.tokens))       # returns number of tokens (document length)
print(len(set(ko.tokens)))  # returns number of unique tokens
ko.vocab()                  # returns frequency distribution

# 5. 챠트로 보기
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
#
# font_fname = 'c:\\windows\\fonts\\malgun.ttf'
# font_name = font_manager.FontProperties(fname=font_fname).get_name()
# rc('font', family=font_name)
#
# plt.figure(figsize=(20,10))
#text.plot(50) #상위 50개의 unique token에 대해서 plot한결과를 아래와 같이 보여준다.

# 6. 특정 단어에 대해서 빈도수 확인하기
print(ko.count(str('스토리')))
print(ko.count(str('영상')))
print(ko.count(str('연출')))
print(ko.count(str('연기')))
print(ko.count(str('아름')))

# 7. 분산 차트 보기 (dispersion plot)
#ko.dispersion_plot(['스토리','영상','연출','연기','아름'])

# 8. 색인하기(본문속에서 해당 검색어가 포함된 문장을 찾아주는 것)
story_tmp = ko.concordance_list(('스토리'))
image_tmp = ko.concordance_list(('영상'))
direct_tmp = ko.concordance_list(('연출'))
act_tmp = ko.concordance_list(('연기'))
me_tmp = ko.concordance_list(('아름'))
# 7. 분산 차트 보기 (dispersion plot)
# ko.dispersion_plot(['스토리','영상','연출','연기','아름'])

# 8. 색인하기(본문속에서 해당 검색어가 포함된 문장을 찾아주는 것)
story_tmp = ko.concordance_list(('스토리'))
image_tmp = ko.concordance_list(('영상'))
direct_tmp = ko.concordance_list(('연출'))
act_tmp = ko.concordance_list(('연기'))
me_tmp = ko.concordance_list(('아름'))

# # 스토리 관련 문장을 추출한다.
# str_story_tmp = ""
# for i, v in enumerate(story_tmp):
#     str_story_tmp = str_story_tmp + story_tmp[i][4]
# str_story_tmp = str_story_tmp.split('\n')
# pprint(str_story_tmp[0])
#
# # 영상 관련 문장을 추출한다.
# str_image_tmp = ""
# for i, v in enumerate(image_tmp):
#     str_image_tmp = str_image_tmp + image_tmp[i][4]
# str_image_tmp = str_image_tmp.split('\n')
# pprint(str_image_tmp[0])
#
# # 연출 관련 문장을 추출한다.
# str_direct_tmp = ""
# for i, v in enumerate(direct_tmp):
#     str_direct_tmp = str_direct_tmp + direct_tmp[i][4]
# str_direct_tmp = str_direct_tmp.split('\n')
# pprint(str_direct_tmp[0])
#
# # 연기 관련 문장을 추출한다.
# str_act_tmp = ""
# for i, v in enumerate(act_tmp):
#     str_act_tmp = str_act_tmp + act_tmp[i][4]
# str_act_tmp = str_act_tmp.split('\n')
# pprint(str_act_tmp[0])
#
# str_me_tmp = ""
# for i, v in enumerate(me_tmp):
#     str_me_tmp = str_me_tmp + me_tmp[i][4]
# str_me_tmp = str_me_tmp.split('\n')
# pprint(str_me_tmp[0])
#
# print("스토리 관련 문장")
# for i in range(len(str_story_tmp)):
#     predict_pos_neg(str_story_tmp[i])
# print("영상 관련 문장")
# for i in range(len(str_image_tmp)):
#     predict_pos_neg(str_image_tmp[i])
# print("연출 관련 문장")
# for i in range(len(str_direct_tmp)):
#     predict_pos_neg(str_direct_tmp[i])
# print("연기 관련 문장")
# for i in range(len(str_act_tmp)):
#     predict_pos_neg(str_act_tmp[i])
# print("아름 관련 문장")
# for i in range(len(str_me_tmp)):
#     predict_pos_neg(str_me_tmp[i])