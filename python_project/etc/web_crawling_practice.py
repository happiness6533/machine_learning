import requests
from bs4 import BeautifulSoup
import time

data = []
for i in range(170000, 170200):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        res = requests.get("https://www.gsshop.com/shop/spa/nowBest.gs?lseq=415303-1&gsid=gnb-AU415303-AU415303-1",
                           headers=headers)
        res.encoding = 'utf8'
        res_html = res.text
        soup = BeautifulSoup(res_html, 'html.parser')
        print(soup)
        get_tag_list = soup.select('#nowBestShop')
        print("제목: ", get_tag_list[0].text)
        data.append(get_tag_list[0].text)
        time.sleep(1)
    except:
        print("데이터가 없는데?")
