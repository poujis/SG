from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import json
import re
import requests
from datetime import datetime
from collections import OrderedDict

# 네이버 검색 Open API 사용 요청시 얻게되는 정보를 입력합니다
naver_client_id = "LwIv4iBS06sNbnFNjYSG"
naver_client_secret = "OVpKPW1ZUd"


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def searchByTitle(title):
    myurl = 'https://openapi.naver.com/v1/search/movie.json?display=100&query=' + quote(title)
    request = urllib.request.Request(myurl)
    request.add_header("X-Naver-Client-Id", naver_client_id)
    request.add_header("X-Naver-Client-Secret", naver_client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        d = json.loads(response_body.decode('utf-8'))
        if (len(d['items']) > 0):
            return d['items']
        else:
            return None

    else:
        print("Error Code:" + rescode)


def findItemByInput(items):
    movie_list = []

    for index, item in enumerate(items):
        navertitle = cleanhtml(item['title'])
        naversubtitle = cleanhtml(item['subtitle'])
        naverpubdate = cleanhtml(item['pubDate'])
        naveractor = cleanhtml(item['actor'])
        naverlink = cleanhtml(item['link'])
        naveruserScore = cleanhtml(item['userRating'])
        naverdirector = cleanhtml(item['director'])

        navertitle1 = navertitle.replace(" ", "")
        navertitle1 = navertitle1.replace("-", ",")
        navertitle1 = navertitle1.replace(":", ",")

        # 기자 평론가 평점을 얻어 옵니다
        spScore = getSpecialScore(naverlink)

        # 네이버가 다루는 영화 고유 ID를 얻어 옵니다다
        naverid = re.split("code=", naverlink)[1]

        # 영화의 타이틀 이미지를 표시합니다
        # if (item['image'] != None and "http" in item['image']):
        #    response = requests.get(item['image'])
        #    img = Image.open(BytesIO(response.content))
        #    img.show()

        # print(index, navertitle, naversubtitle, naveruserScore, spScore)

        movie_list.append(index)
        movie_list.append(navertitle)
        movie_list.append(naversubtitle)
        movie_list.append(naveruserScore)
        movie_list.append(spScore)
        movie_list.append(director)

    # print(movie_list)

    return movie_list


def getInfoFromNaver(searchTitle):
    items = searchByTitle(searchTitle)
    movie_list_tmp = []

    if (items != None):
        movie_list_tmp = findItemByInput(items)
    else:
        print("No result")

    return movie_list_tmp

def get_soup(url):
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')
    return soup


# 기자 평론가 평점을 얻어 옵니다
def getSpecialScore(URL):
    soup = get_soup(URL)
    scorearea = soup.find_all('div', "spc_score_area")
    newsoup = BeautifulSoup(str(scorearea), 'lxml')
    score = newsoup.find_all('em')
    if (score and len(score) > 5):
        scoreis = score[1].text + score[2].text + score[3].text + score[4].text
        return float(scoreis)
    else:
        return 0.0


def get_movie_review_data(url):
    movie_review_data = []
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
    for li in lis:
        nickname = li.findAll('a')[0].find('span').getText()
        created_at = datetime.strptime(li.find('dt').findAll('em')[-1].getText(), "%Y.%m.%d %H:%M")

        review_text = li.find('p').getText()
        score = li.find('em').getText()
        btn_likes = li.find('div', {'class': 'btn_area'}).findAll('span')
        like = btn_likes[1].getText()
        dislike = btn_likes[3].getText()

        watch_movie = li.find('span', {'class': 'ico_viewer'})

        # 간단하게 프린트만 했습니다.
        # print(nickname, review_text, score, like, dislike, created_at, watch_movie and True or False)
        movie_review_data.append(nickname)
        movie_review_data.append(review_text)
        movie_review_data.append(score)
        movie_review_data.append(like)
        movie_review_data.append(dislike)
        movie_review_data.append(created_at)

    return movie_review_data

def get_movie_review(code):
    test_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=' +code+ '&type=after'
    resp = requests.get(test_url)
    html = BeautifulSoup(resp.content, 'html.parser')
    result = html.find('div', {'class': 'score_total'}).find('strong').findChildren('em')[1].getText()
    total_count = int(result.replace(',', ''))

    # 너무 많다.. 조금만
    total_count = (total_count * 3) / 100

    movie_review_data = []
    for i in range(1, int(total_count / 10) + 1):
        url = test_url + '&page=' + str(i)
        print('url: "' + url + '" is parsing....')
        movie_review_data.append(get_movie_review_data(url))

    return(movie_review_data)

if __name__ == "__main__":
    print(getInfoFromNaver(u"The Mummy"))
    get_movie_review('136990')