# 1. 읽기
#!/usr/bin/env python
# -- coding: utf-8 --
import sys
import importlib
importlib.reload(sys)

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
# 스토리, 비주얼, 연출, 연기
ko = nltk.Text(tokens_ko_morphs, name='연기')   # 이름은 아무거나

# 4. 토근 정보및 단일 토큰 정보 알아내기
print(len(ko.tokens))       # returns number of tokens (document length)
print(len(set(ko.tokens)))  # returns number of unique tokens
ko.vocab()                  # returns frequency distribution

# 5. 챠트로 보기
# ko.plot(50) #상위 50개의 unique token에 대해서 plot한결과를 아래와 같이 보여준다.

# 6. 특정 단어에 대해서 빈도수 확인하기
print(ko.count(str('연기')))

# 7. 분산 차트 보기 (dispersion plot)
# ko.dispersion_plot(['스토리','영상','연출','연기'])

# 8. 색인하기(본문속에서 해당 검색어가 포함된 문장을 찾아주는 것)
ko.concordance(('스토리'))
ko.concordance(('영상'))
ko.concordance(('연출'))
ko.concordance(('연기'))

# 9. 유의어 찾기
ko.similar('스토리')
ko.similar('영상')
ko.similar('연출')
ko.similar('연기')

# 10. 의미단위로 나누기(Tagging and chunking)
# 10.1 형태소 분석기(POS tagging)
# 명사, 형용사, 조사등을 나누어 보여준다.
tags_ko = okt.pos('작고 노란 강아지가 고양이에게 짖었다')

print(tags_ko)

# 10.2 명사구단위로 묵어주기(Noun Phrase Chunking)
parser_ko = nltk.RegexpParser("NP: {<Adjective>*<Noun>*}")
chunks_ko = parser_ko.parse(tags_ko)

chunks_ko.draw()


# 3.1. Topic Modeling (LSI,LDA,HDP 알고리즘)
#!/usr/bin/env python
# -- coding: utf-8 --
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from konlpy.corpus import kobill
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]

from konlpy.tag import Twitter; t=Twitter()
pos = lambda d: ['/'.join(p) for p in t.pos(d, stem=True, norm=True)]
texts_ko = [pos(doc) for doc in docs_ko]

#encode tokens to integers
from gensim import corpora
dictionary_ko = corpora.Dictionary(texts_ko)
dictionary_ko.save('ko.dict')  # save dictionary to file for future use

#calulate TF-IDF
from gensim import models
tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]
corpora.MmCorpus.serialize('ko.mm', tfidf_ko) # save corpus to file for future use

#train topic model
#LSI
ntopics, nwords = 3, 5
lsi_ko = models.lsimodel.LsiModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
print(lsi_ko.print_topics(num_topics=ntopics, num_words=nwords))

#LDA
import numpy as np; np.random.seed(42)  # optional
lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
print(lda_ko.print_topics(num_topics=ntopics, num_words=nwords))

#HDP
import numpy as np; np.random.seed(42)  # optional
hdp_ko = models.hdpmodel.HdpModel(tfidf_ko, id2word=dictionary_ko)
print(hdp_ko.print_topics(topics=ntopics, topn=nwords))

#Scoring document
bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[0])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[1])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)