
import requests
import json
import time



num=1
url = "http://api.springernature.com/meta/v2/json?q=(subject:%22Material%20Science%22 OR subject:%22Physics%22 OR subject:%22Chemistry%22 AND year:2000)&p=100&s={}&api_key=75d857558a127c1334ab4867894f776e"
path= r'F:\zhuomian\数据库\2000.txt'
def RESULT(URL_1):
    URL_1=URL_1.format(str(num))
    response = requests.get(URL_1)
    response = json.loads(response.text)
    result = response.get('result')
    print(result)


RESULT(url)


while (1):
            URL_2= "http://api.springernature.com/meta/v2/json?q=(subject:%22Material%20Science%22 OR subject:%22Physics%22 OR subject:%22Chemistry%22 AND year:2016)&p=100&s={}&api_key=75d857558a127c1334ab4867894f776e".format(str(num))
            try:
                response = requests.get(URL_2)
                response = json.loads(response.text)
                Abstract = response.get('records')
                for i in Abstract:
                 f = open(path, 'a', encoding='UTF8')
                 a = i.get('abstract')
                 print(a, file=f)
                 num = num + 100
                 print(num)


            except:
                     print('错误')
