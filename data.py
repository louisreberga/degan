from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os
import requests
import time

option = webdriver.ChromeOptions()
option.add_argument("headless")
service = Service("./chromedriver")
driver = webdriver.Chrome(service=service, options=option)

driver.get("https://howrare.is")
name = 'DeGods'
os.mkdir('data/DeGods')
href = 'https://howrare.is/degods'
page = 0
length = 1

while lenght > 0:
    driver.get(f"{href}/?page={page}")
    nfts = driver.find_elements(by=By.XPATH, value="//div[@class='nft_item_img']//a")
    links = [nft.get_attribute("href") for nft in nfts]
    lenght = len(nfts)

    for link in links:
        print(link)
        driver.get(link)
        image = driver.find_element(by=By.XPATH, value="//div[@class='nfts_detail_img']//a//img")
        src = image.get_attribute("src")

        img = requests.get(src).content
        with open('data/DeGods/' + link.split('/')[-2] + '.' + src.split('.')[-1], 'wb') as handler:
            handler.write(img)

        print(link.split('/')[-2])
        driver.back()
        time.sleep(0.5)

    driver.back()
    page += 1
