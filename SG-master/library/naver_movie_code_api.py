
from bs4 import BeautifulSoup

import urllib.request
from urllib import parse

maximum = 0
maxpage = 40
maxpage_t =(int(maxpage)-1)*10+1

movie_code_list = []

def get_movie_code(title_tmp):
    page = 1

    while page <= maxpage_t:

        html = urllib.request.urlopen('https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&page=' + str(page))
        soup = BeautifulSoup(html, 'lxml')
        titles = soup.find_all('td', 'title')
        front_url = 'http://movie.naver.com/'

        for title in titles:
            url = parse.urlparse(front_url + title.find('a')['href'])
            key_tmp = parse.parse_qs(url.query)
            key_tmp['title'] = title.find('a').text

            if(key_tmp['title'] == title_tmp):
                # print(title_tmp, key_tmp['code'])
                return key_tmp['code']

            # print(key_tmp)
            movie_code_list.append(key_tmp)

            # matching = [s for s in movie_code_list if "그린 북" in s]

        page += 10

    return movie_code_list

if __name__ == "__main__":
    print(get_movie_code('여고괴담'))