# Author: BBK
# Date: 2022/1/22 

from selenium import webdriver
from selenium.webdriver.support.select import Select
import time 
import re 
import numpy as np 
import pandas as pd 
import os 

def data_processing(table_text):
    cols = []
    data = []
    for i in table_text.split("\n"):
        if (u'合' not in i) and (u'計' not in i): 
            if not re.search("\d", i):
                c = i.split()
                if len(c) == 5:
                    cols.append(c)
                elif len(c) == 4:
                    cols.append(['單位'] + c )
            else:
                b = i.split()
                temp = []
                for j in b:
                    if re.search('\,', j):
                        a = float(re.sub("\,","",j))                     
                    elif re.search('\.', j) or re.search('\d', j):
                        a = float(j)
                    else:
                        a = j                
                    temp.append(a)
                data.append(temp)
        else:
            pass 
    return data, cols 

def text_convert(b_text):
    temp1 = []
    temp2 = []
    for i in b_text.split("\n"):
        t  = re.sub('\s', '', i)
        temp1.append(t)
        k = ''
        for g in re.findall("[^\d.]",t):
            k += g
        temp2.append(k)
    return dict(zip(temp2, temp1))

def convert_to_dataframe(data, cols, result):
    cols = pd.DataFrame(cols)
    data = pd.DataFrame(data)
    result[year] = pd.concat([cols, data], axis = 0)
    g = result[year].iloc[0]
    h = result[year].iloc[1]
    f = pd.Series(['_']*len(h))
    header = g+f+h
    result[year].columns = header 
    result[year].index = np.arange(result[year].shape[0]) 
    result[year].drop([0,1], axis = 0, inplace = True )
    result[year].set_index(result[year].columns[0], inplace = True)
    return result 

def summary(province, years, **result_dict):
    df_summary = pd.DataFrame(index = province, columns = years)

    for pro in province:
        for year in years:
            try:
                temp_df = result_dict[year]
                a = temp_df[temp_df.index == pro]['收量_公斤'][0]
                df_summary.loc[pro,year] = a
            except:
                pass 
    df_summary['收量_公斤'] = np.sum(df_summary,axis = 1)
    return df_summary



if __name__ == '__main__':

    start = int(input("起始年分(ex: 101):"))
    end = int(input("終止年分(ex: 109):"))
    crop = input("作物名稱(ex: 小麥):")
    browser = input("所使用瀏覽器(ex: google or edge):")
    
    province = []
    years = [str(y) for y in np.arange(start,end+1)]
    result_dict = {}
    
    # driver = webdriver.Edge(executable_path="C:\\Users\\bbkelly\\Desktop\Kelly\\python_package_no_pip\\selenium\\msedgedriver.exe")
    browsers = {"edge":".\\msedgedriver.exe", "google": ".\\chromedriver.exe"}
    if browser == 'edge':
        driver = webdriver.Edge(executable_path=browsers[browser])
    elif browser == 'google':
        driver = webdriver.Chrome(executable_path=browsers[browser])
    
    driver.implicitly_wait(0.5)
    driver.get("https://agr.afa.gov.tw/afa/afa_frame.jsp")
    driver.switch_to.frame("left_frame")
    link = driver.find_element_by_xpath('//*[@id="divoFoldMenu0_0_1"]/a').get_attribute("href")
    driver.get(link)
    
    try:
        for year in years:
            select = Select(driver.find_element_by_xpath("/html/body/div/form/div/table/tbody/tr[1]/td[2]/select"))
            select.select_by_visible_text(year)
            
            select = Select(driver.find_element_by_xpath("/html/body/div/form/div/table/tbody/tr[2]/td[2]/select"))
            select.select_by_visible_text('03.全年作')
            
            b = driver.find_element_by_xpath("/html/body/div/form/div/table/tbody/tr[3]/td[2]/select")
            b_text = b.text
            convert = text_convert(b_text)
            select = Select(b)
            select.select_by_visible_text(convert[crop])
            
            select = Select(driver.find_element_by_xpath("/html/body/div/form/div/table/tbody/tr[4]/td[2]/select"))
            select.select_by_visible_text('00.全部')
            
            driver.find_element_by_xpath("/html/body/div/form/div/table/tbody/tr[5]/td[2]/input[1]").click()
            table = driver.find_element_by_xpath("/html/body/div/form/div/table")
        
            table_text = table.text
            data, cols = data_processing(table_text)
            result_dict = convert_to_dataframe(data, cols, result_dict)
            province = province + list(result_dict[year].index)
            driver.back()
            time.sleep(0.1)
    except:
        print("資料期間只到109年!") 
    driver.close()
    
    province = list(set(province))
    df_summary = summary(province, years, **result_dict)
    with pd.ExcelWriter('農情網爬蟲結果.xlsx') as writer: 
        for key, value in result_dict.items(): 
            value.to_excel(writer, sheet_name=key)
        df_summary.to_excel(writer, sheet_name=u'總計')
        path = os.getcwd() + '\\農情網爬蟲結果.xlsx'
        print("[執行結果] 檔案 '農情網爬蟲結果.xlsx' 已儲存在: {}".format(path))
    