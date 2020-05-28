# -*- coding: utf-8 -*-
"""
Created on Sun May 7 02:49:32 2020

@author: Jayleen

Crawling from https://www.wunderground.com/history/daily/us/ca/san-francisco/KSFO
"""
#%% full

from selenium import webdriver
import time
from bs4 import BeautifulSoup
import pandas as pd
import os

days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

driver = webdriver.Chrome("./chromedriver")

for year in range(2003, 2020):
    for month in range(1, 13):
        if (year == 2004 or year == 2008 or year == 2012 or year == 2016 or year == 2020):
            days[1] = 29
        else:
            days[1] = 28
        for day in range(1, days[month-1]+1):
            print(year, month, day)
            driver.get("https://www.wunderground.com/history/daily/us/ca/san-francisco/KSFO/date/%d-%d-%d" %(year, month, day)) #헤더 필요없음
            driver.implicitly_wait(9)

            try:
                setting = driver.find_element_by_css_selector("#wuSettings> i")
                setting.click()
                to_C = driver.find_element_by_css_selector("#wuSettings-quick > div > a:nth-child(2)")
                to_C.click()
                time.sleep(3)

            except:
                pass

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            time = soup.select("td.mat-cell.cdk-cell.cdk-column-dateString")
            temperature = soup.select("td.mat-cell.cdk-cell.cdk-column-temperature")
            dewpoint = soup.select("td.mat-cell.cdk-cell.cdk-column-dewPoint")
            humidity = soup.select("td.mat-cell.cdk-cell.cdk-column-humidity")
            wind = soup.select("td.mat-cell.cdk-cell.cdk-column-windcardinal")
            wind_speed = soup.select("td.mat-cell.cdk-cell.cdk-column-windSpeed")
            wind_gust = soup.select("td.mat-cell.cdk-cell.cdk-column-windGust")
            pressure = soup.select("td.mat-cell.cdk-cell.cdk-column-pressure")
            precip = soup.select("td.mat-cell.cdk-cell.cdk-column-precipRate")
            condition = soup.select("td.mat-cell.cdk-cell.cdk-column-condition")

            t_list = []
            c_list = []
            d_list = []
            h_list = []
            w_list = []
            ws_list = []
            wg_list = []
            p_list = []
            pc_list = []
            cd_list = []

            for t in time:
                t = t.text.replace(' ', '')
                t_list.append(t)

            for c in temperature:
                c = int(c.text.replace('C', '').strip())
                c_list.append(c)

            for d in dewpoint:
                d = int(d.text.replace('C', '').strip())
                d_list.append(d)

            for h in humidity:
                h = int(h.text.replace('\xa0%', '').strip())
                h_list.append(h)

            for w in wind:
                w = w.text.strip()
                w_list.append(w)

            for w in wind_speed:
                w = int(w.text.replace("\xa0km/h", '').strip())
                ws_list.append(w)

            for w in wind_gust:
                w = int(w.text.replace("\xa0km/h", '').strip())
                wg_list.append(w)

            for pr in pressure:
                pr = float(pr.text.replace("\xa0hPa", '').replace(",","").strip())
                p_list.append(pr)

            for pc in precip:
                pc = float(pc.text.replace('\xa0mm', '').strip())
                pc_list.append(pc)

            for cd in condition:
                cd = cd.text.strip()
                cd_list.append(cd)


            df = pd.DataFrame({"Date": ["%d-%d-%d" %(year,month,day)]*len(time), "Time" : t_list, "Temperature(˚C)" : c_list, 
                                    "DewPoint(˚C)" : d_list, 'Humidity(%)' : h_list, 'Wind': w_list, 
                                    'Wind_speed(km/h)' : ws_list, 'Wind_gust': wg_list, 'Pressure(hPa)' : p_list, 
                                    "PrecipRate(mm)" : pc_list, 'Condition' : cd_list}) 

            print(df.Date[0], "FINISH")

            if not os.path.exists('SFO_Weather_2003_to_present2.csv'):
                        df.to_csv('SFO_Weather_2003_to_present2.csv', index=False, mode='w', encoding='utf-8')
            else:
                        df.to_csv('SFO_Weather_2003_to_present2.csv', index=False, mode='a', encoding='utf-8', header=False)

