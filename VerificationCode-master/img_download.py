# -*- coding: utf-8 -*-
# @Time  : 2021/3/18 22:46
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================
import os

def img_download():
    path = "./data/new"
    if(not os.path.exists(path)):
        os.makedirs(path)
    import requests
    headers = {
        "Cookie": "XXXXXXXXXXXXXX",
        "Referer": "https://user.ichunqiu.com/register/index",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"

    }
    url = "https://user.ichunqiu.com/login/verify_image?d=1523697519026"
    for i in range(0, 20):
        with open("{}/{}.png".format(path,i), "wb") as f:
            print("{} pic downloading...".format(i))
            f.write(requests.get(url, headers=headers).content)


if(__name__ == "__main__"):
    img_download()