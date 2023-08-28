from flask import Flask, redirect, url_for, request, render_template, session, jsonify #Flask
import json 
import pandas as pd
import sqlite3
import json
import datetime
import numpy as np
import os
from os.path import join, getsize
import time
from scipy import stats
import sys
import traceback
from dateutil.parser import parse
import functools as ft
import zipfile
import shutil
import gc
import glob
import re
#單query數量= x/3-11
sys.setrecursionlimit(3500)

#api被呼叫的當下日期
global today_date
today_date = "'"+time.strftime("%Y-%m-%d 00:00:00")+"'"
global ex_df
ex_df= pd.DataFrame()
#設置表格style參數
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option('precision', 0)
# pd.set_option("display.float_format",lambda x : '%.0f' % x)
pd.options.mode.chained_assignment = None

def logic_json(logic):
    ##############################################################
    global ID_list_sult

    with open(r''+logic) as f:
        logic_structure = json.load(f)
    
    sort = logic_structure.get("table_select")
    i=0
    global table_list
    table_list = []
    for i in range(len(sort)):
        table_list.append(sort[i]['table'])
        i=i+1
            
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor    
    ############################# CRLF #################################
    df_CRLF = pd.read_sql("SELECT * FROM " + "CRLF", conn)
    df_CRLF['age'] = df_CRLF['age'].astype(int)
    df_CRLF = df_CRLF.query(logic_structure['disease'],engine='python')
    
    p = table_list.index('CRLF')
    i=0
    for each_query in range(len(logic_structure['table_select'][p]['queryList'])):
        each = list(logic_structure['table_select'][p]['queryList'].values())[i][0]
        try:
            df_CRLF = df_CRLF.query(each)
        except:
            return([''])
        i+=1
    ##############################################################   
    ID_list = df_CRLF['id'].tolist()
    ID_list_sult=[]
    i=0
    for make_cond in range(len(ID_list)):
        if (i+1)!=len(ID_list):
            ID_list_sult.append('id=='+"'"+ID_list[i]+"'"+'|')
            i+=1
        else:
            ID_list_sult.append('id=='+"'"+ID_list[i]+"'")
            i+=1

    ID_list_sult = "".join(map(str, ID_list_sult))    
    ##############################################################
    
    conn.close()
    return(ID_list_sult)

def age_group(r):

    try:
        r=int(r)
    except:
        r="999"
        return r

    if r<15:
        r="<15"
        return r
    elif r>=15 and r<=29:
        r="15-29"
        return r
    elif r>=30 and r<=49:
        r="30-49"
        return r
    elif r>=50 and r<=64:
        r="50-64"
        return r
    elif r>=65 and r<=74:
        r="65-74"
        return r
    elif r>=75 and r!=999:
        r=">74"
        return r
    else:
        r="999"
        return r
    try:
        dataframe['age_group'] = dataframe['age'].apply(age_group)
    except:
        pass

def fourdate(x):
    try:
        x = str(x)
        x = x[0:4]
        return x
    except:
        x = "9999"
        return x

def insert_age_gender(df_summary):
    def cut_date(x):
        try:          
            if len(x)>8:
                x = x[0:8]
            x_f = x[0:4]
            x_f = x_f.replace('9999','1900')
            x_m = x[4:6]
            x_m = x_m.replace('99','01')
            x_b = x[6:8]
            x_b = x_b.replace('99','01')
            x = x_f + x_m + x_b
            x = parse(x)
            return x

        except:
            x_error = '19000101'
            x_error = parse(x_error)
            return x_error

    #DB conn
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor

    #read summary
    df_summary_IDlist = df_summary['id'].tolist()
    df_summary_IDlist

    #select_正規寫法, 現id、gender不動可丟表, 若未來浮動掉, 需兩步驟撈max->取index->撈index, 如下
    #insert_TOTFAE = insert_TOTFAE[['id','age_mix','gender']].max(skipna=True).to_frame().T 替換處
    # data={
    # 'id':['小明','小王','小美'],
    # 'age_mix':[55,33,44],
    # 'gender':[1,1,2]}
    # df=pd.DataFrame(data)
    # print(df)

    # print('------------------')

    # ind = df["age_mix"].idxmax()
    # row = df.loc[ind,:].to_frame().T
    # print(row)

    #select_AE
    combine_TOTFAE = pd.DataFrame({'id':[],'age_mix':[]})
    i=0
    for ae in range(len(df_summary_IDlist)):
        try:
            insert_TOTFAE = pd.read_sql("SELECT d3,d9,d11,gender FROM " + "[" + 'TOTFAE' + "]" + "where d3=="+"'"+df_summary_IDlist[i]+"'" + "COLLATE NOCASE", conn)
            insert_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
            insert_TOTFAE['d11'] = insert_TOTFAE['d11'].apply(cut_date)
            insert_TOTFAE['d9'] = insert_TOTFAE['d9'].apply(cut_date)
            insert_TOTFAE['age_mix'] = (insert_TOTFAE['d9'].dt.year) - (insert_TOTFAE['d11'].dt.year)
            insert_TOTFAE = insert_TOTFAE[['id','age_mix','gender']].max(skipna=True).to_frame().T
            combine_TOTFAE = pd.concat([combine_TOTFAE,insert_TOTFAE], axis=0, ignore_index=True)
            i+=1
            
        except:
            i+=1
            pass

    #select_BE
    combine_TOTFBE = pd.DataFrame({'id':[],'age_mix':[]})
    i=0
    for be in range(len(df_summary_IDlist)):
        try:
            insert_TOTFBE = pd.read_sql("SELECT d3,d10,d6,gender FROM " + "[" + 'TOTFBE' + "]" + "where d3=="+"'"+df_summary_IDlist[i]+"'" + "COLLATE NOCASE", conn)
            insert_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
            insert_TOTFBE['d10'] = insert_TOTFBE['d10'].apply(cut_date)
            insert_TOTFBE['d6'] = insert_TOTFBE['d6'].apply(cut_date)
            insert_TOTFBE['age_mix'] = (insert_TOTFBE['d10'].dt.year) - (insert_TOTFBE['d6'].dt.year)
            insert_TOTFBE = insert_TOTFBE[['id','age_mix','gender']].max(skipna=True).to_frame().T
            combine_TOTFBE = pd.concat([combine_TOTFBE,insert_TOTFBE], axis=0, ignore_index=True)
            i+=1
            
        except:
            i+=1
            pass

    #select_DE
    combine_TOTFDE = pd.DataFrame({'id':[],'age_mix':[]})
    i=0
    for de in range(len(df_summary_IDlist)):
        try:
            insert_TOTFDE = pd.read_sql("SELECT id,d4,d3,gender FROM " + "[" + 'DEATH' + "]" + "where id=="+"'"+df_summary_IDlist[i]+"'" + "COLLATE NOCASE", conn)
            insert_TOTFDE['d4'] = insert_TOTFDE['d4'].apply(cut_date)
            insert_TOTFDE['d3'] = insert_TOTFDE['d3'].apply(cut_date)
            insert_TOTFDE['age_mix'] = (insert_TOTFDE['d4'].dt.year) - (insert_TOTFDE['d3'].dt.year)
            insert_TOTFDE = insert_TOTFDE[['id','age_mix','gender']].max(skipna=True).to_frame().T
            combine_TOTFDE = pd.concat([combine_TOTFDE,insert_TOTFDE], axis=0, ignore_index=True)
            i+=1
            
        except:
            i+=1
            pass
    try: 
        # groupby
        age_mix_df = pd.concat([combine_TOTFAE,combine_TOTFBE,combine_TOTFDE], axis=0, ignore_index=True)

        gender_df = age_mix_df.groupby('id')['gender'].max().to_frame().reset_index()
        age_df = age_mix_df.groupby('id')['age_mix'].max().to_frame().reset_index()

        age_mix_df = pd.merge(age_df, gender_df, how='left', on=['id'], indicator=False).fillna(np.nan).replace([np.nan],[None]) #篩掉非癌登ID冊者減少人數
        age_mix_df

        # merge_summary
        df_summary2 = pd.merge(df_summary, age_mix_df, how='left', on=['id'], indicator=False).fillna(np.nan).replace([np.nan],[None]) #篩掉非癌登ID冊者減少人數  
        df_summary2['age_mix'] = df_summary2['age_mix'].astype(str)
        df_summary2.loc[df_summary2['age'].isnull(),'age'] = df_summary2.loc[df_summary2['age_mix'].notnull(),'age_mix']
        df_summary2.loc[df_summary2['sex'].isnull(),'sex'] = df_summary2.loc[df_summary2['gender'].notnull(),'gender']
        # age_group again
        # try:
        #     df_summary2['age'] = df_summary2['age'].astype(int)
        # except:
        #     pass
        df_summary2['age_group'] = df_summary2['age'].apply(age_group)

    except:
        
        conn.close()
        return(df_summary) #預防完全找不到gender or age_mix
    
    conn.close()
    return(df_summary2)

def missing(dataframe_des, dataframe):
    try:
        dataframe_missing = dataframe.isnull().sum()
        dataframe_missing = dataframe_missing.to_frame('missing')
        concat_df = pd.concat([dataframe_des, dataframe_missing],axis=1)
        # concat_df['missing'] = concat_df['missing'].fillna(0)
    except:
        dataframe_missing = pd.DataFrame(columns=['missing'])
        concat_df = pd.concat([dataframe_des, dataframe_missing],axis=1)
        # concat_df['missing'] = concat_df['missing'].fillna(0)

    return(concat_df)

def summ(dataframe_des, dataframe):
    dataframe = dataframe.drop_duplicates(subset = ["id"])
    df_sum = dataframe.agg(sum)
    df_sum = df_sum.to_frame()
    df_sum.rename(columns={0:'sum'}, inplace = True)
    df_sum = df_sum.T
    concat_df = pd.concat([df_sum, dataframe_des],axis=0)
    concat_df = concat_df.dropna(axis=1,how='any')
    return(concat_df)
        
def split_v_c(dataframe_des):
    
    try:
        dataframe_des_v = dataframe_des[['count','mean','std','min','25%','50%','75%','max']].dropna(axis=0,how='any')
    except:
        dataframe_des_v = pd.DataFrame(columns=['count','mean','std','min','25%','50%','75%','max'])

    try:
        dataframe_des_c = dataframe_des[['count','unique','top','freq','missing']].dropna(axis=0,how='any')
    except:
        dataframe_des_c = pd.DataFrame(columns=['count','unique','top','freq','missing'])
    
    return(dataframe_des_v,dataframe_des_c)

def describe(df):
    try:
        df['age'] = df['age'].astype(int)
    except:
        pass
    try:
        df['d14'] = df['d14'].astype(int)
        df['d15'] = df['d15'].astype(int)
    except:
        pass
    df_des = df.describe(include='all',datetime_is_numeric=True)
    return df_des

def cross(df):
    try:
        c = pd.crosstab(df[x], df[y], margins=True, margins_name='Total') #[0]欄位類別 [1]統計變數
        c = c.T
        #
        c = c.astype(str) #c是計次
        # num_1 = list(c['1'])[-1] + "(" + str( round( int(list(c['1'])[-1])/int(list(c['Total'])[-1])*100,2 ) ) +"%" + ")" 
        # num_2 = list(c['2'])[-1] + "(" + str( round( int(list(c['2'])[-1])/int(list(c['Total'])[-1])*100,2 ) ) +"%" + ")"
    #   --------------------------------------------------------------------------------------------------------------
        p = pd.crosstab(df[x], df[y], normalize='columns')
        p = p.apply(lambda x: x*100)
        p = p.T.round(1)
        p = p.astype(str)
        p = p.apply(lambda x: "(" + x + "%" + ")") #p是(百分比)
    #   --------------------------------------------------------------------------------------------------------------    
        df_right = c.merge(p, how='left', left_index=True, right_index=True)

        tag = [tag for tag in c]
        i=0
        for combine in range(len(tag)-1): #計算欄位類別有幾種(性別兩種，血型可能五種之類) ; 最後一個是total扣掉一個位置
                 df_right[tag[i]] = df_right[str(tag[i])+'_x'] + df_right[str(tag[i]) + '_y'] #c、p放在一起 2(50%)這樣
                 i+=1
        df_right = df_right[tag]  # 僅留下合併後的;分開的不keep

        
        if '1' in df_right.columns:
            pass
        else:
            df_right['1']=0

        if '2' in df_right.columns:
            pass
        else:
            df_right['2']=0 

        df_right.rename(columns = {'1':'male'}, inplace = True)
        df_right.rename(columns = {'2':'female'}, inplace = True)
        df_right=df_right[:-1]
        return(df_right)

    except:
        df_right = pd.DataFrame(columns=['var','group1','group2','total']) 
        df_right=df_right[:-1]
        return(df_right)

def Bsingle_demo(table,x,y,stats,ID,logic):
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()

    # today_date = "'"+time.strftime("%Y-%m-%d 00:00:00")+"'"
    table_name = str(table)
    df = pd.read_sql("SELECT Max(ModifyTime) FROM " + "[" + table_name + "]" , conn)
    df['Max(ModifyTime)']= pd.to_datetime(df['Max(ModifyTime)']) - pd.Timedelta(days=1)
    df = df.astype(str)
    today_date = "'"+df.iat[0,0]+"'"


    if len(logic)>3:

        logic_query = logic_json(logic)

        if len(logic_query)<3:
            return(ex_df,ex_df,ex_df,ex_df)

    #sql to pandas
    global dataframe
    if len(ID)>=1:
        try:
            dataframe  = pd.read_sql("SELECT * FROM " + "[" + table + "]" + "where id=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        except:
            pass
        try:
            dataframe  = pd.read_sql("SELECT * FROM " + "[" + table + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        except:
            pass
        try:
            dataframe  = pd.read_sql("SELECT * FROM " + "[" + table + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        except:
            pass
    else:
        try:
            dataframe  = pd.read_sql("SELECT * FROM " + "[" + table + "]" , conn)
        except:
            pass

    #date_type change
    try:
        dataframe['CreateTime'] = dataframe['CreateTime'].astype('datetime64[ns]')
        dataframe['ModifyTime'] = dataframe['ModifyTime'].astype('datetime64[ns]')
    except:
        pass

    #summary table type change
    # try:
    #     dataframe['ModifyTime'] = dataframe['ModifyTime'].astype('datetime64[ns]')
    # except:
    #     pass

    #drop col
    try:
        dataframe.drop('verify',axis=1,inplace=True)
    except:
        pass
    try:
        dataframe.drop('IsUploadHash',axis=1,inplace=True)
    except:
        pass
    try:
        dataframe.drop('Index',axis=1,inplace=True)
    except:
        pass
    try:
        dataframe.rename(columns={'h9':'id'}, inplace = True)
    except:
        pass

    if table != 'DEATH':
        if table != 'CASE':
            try:
                dataframe.rename(columns={'d3':'id'}, inplace = True)
            except:
                pass

    if len(logic) >= 2:
        # logic_query = logic_json(logic) #一開始為了判斷篩不到人跑過了，不二跑

        match_df = pd.DataFrame()
        query_batch = logic_query.split('|')

        i=0
        for b in range(len(query_batch)):
            df = dataframe.query(query_batch[i])
            match_df = pd.concat([match_df, df], axis=0)
            i=i+1

        dataframe = match_df

        # dataframe = dataframe.query(logic_query)
    else:
        pass
    # try:
    #     dataframe = dataframe.drop_duplicates(subset = ["id","sequence","site","didiag"])
    # except:
    #     pass
 ################################## space to null for missing ###################################################################

    dataframe = dataframe.replace(r'^\s*$', np.nan, regex=True)

 ################################## date_query ###################################################################
    if stats=='update':
        try:
            dataframe_update = dataframe.copy()
            dataframe_update = dataframe_update.query("ModifyTime >=" + today_date)
        except:
            pass

  ############################ describe ######################################################################################
    if stats=='update':
        des_dataframe_update  = describe(dataframe_update).T
        des_dataframe_update  = missing(des_dataframe_update, dataframe_update)
        des_dataframe_update_v  = split_v_c(des_dataframe_update)[0]
        # des_dataframe_update_v = summ(des_dataframe_update_v.T,dataframe_update).T #時間數字代號等等會被字串sum
        des_dataframe_update_c  = split_v_c(des_dataframe_update)[1]
    else:
        des_dataframe = describe(dataframe).T
        des_dataframe  = missing(des_dataframe, dataframe)
        des_dataframe_v  = split_v_c(des_dataframe)[0]
        # des_dataframe_v  = summ(des_dataframe_v.T,dataframe).T #時間數字代號等等會被字串sum
        des_dataframe_c  = split_v_c(des_dataframe)[1]


  ############################ cross ######################################################################################
    if stats =='update':
        cross_dataframe_update = cross(dataframe_update)

    else:
        cross_dataframe = cross(dataframe)

    if stats=='update':

        
        conn.close()
        return(des_dataframe_update_v,des_dataframe_update_c,cross_dataframe_update,dataframe_update)
    else:
        
        conn.close()  
        return(des_dataframe_v,des_dataframe_c,cross_dataframe,dataframe)
 ################################################################################
 #################################################################################

def B_plus_form2(ID,stats):
    # today_date = "'"+time.strftime("%Y-%m-%d 00:00:00")+"'"
    def max_date(table):
        try:
            table_name = str(table)
            df = pd.read_sql("SELECT Max(ModifyTime) FROM " + "[" + table_name + "]" , conn)
            df['Max(ModifyTime)']= pd.to_datetime(df['Max(ModifyTime)']) - pd.Timedelta(days=1)
            df = df.astype(str)
            today_date = "'"+df.iat[0,0]+"'"
            return(today_date)
        except:
            return("'"+time.strftime("%Y-%m-%d 00:00:00")+"'")

    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()

    today_date_CASE = max_date('CASE')
    today_date_CRLF = max_date('CRLF')
    # today_date_CRSF = max_date('CRSF')
    today_date_DEATH = max_date('DEATH')
    # today_date_LABD1 = max_date('LABD1')
    # today_date_LABD2 = max_date('LABD2')
    today_date_LABM1 =  max_date('LABM1')
    today_date_LABM2 =  max_date('LABM2')
    today_date_TOTFAE = max_date('TOTFAE')
    today_date_TOTFAO1 = max_date('TOTFAO1')
    today_date_TOTFAO2 = max_date('TOTFAO2')
    today_date_TOTFBE = max_date('TOTFBE')
    today_date_TOTFBO1 = max_date('TOTFBO1')
    today_date_TOTFBO2 = max_date('TOTFBO2')

    start = time.perf_counter()
    
    global df_CASE, df_CRLF, df_CRSF, df_DEATH, df_LABD1, df_LABD2, df_LABM1, df_LABM2, df_TOTFAE, df_TOTFAO1, df_TOTFAO2, df_TOTFBE, df_TOTFBO1, df_TOTFBO2
    
    if stats =='update':
        df_CASE = pd.read_sql("SELECT * FROM " + "[" + 'CASE' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_CASE , conn)
        df_CRLF = pd.read_sql("SELECT * FROM " + "[" + 'CRLF' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_CRLF , conn)     
        # df_CRSF = pd.read_sql("SELECT * FROM " + "[" + 'CRSF' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_CRSF , conn)
        df_DEATH = pd.read_sql("SELECT * FROM " + "[" + 'DEATH' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_DEATH , conn)
        # df_LABD1 = pd.read_sql("SELECT * FROM " + "[" + 'LABD1' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_LABD1 , conn)
        # df_LABD2 = pd.read_sql("SELECT * FROM " + "[" + 'LABD2' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_LABD2 , conn)
        df_LABM1 = pd.read_sql("SELECT * FROM " + "[" + 'LABM1' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_LABM1 , conn)
        df_LABM2 = pd.read_sql("SELECT * FROM " + "[" + 'LABM2' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_LABM2 , conn)
        df_TOTFAE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAE' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFAE , conn)
        df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO1' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFAO1 , conn)
        df_TOTFAO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO2' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFAO2 , conn)
        df_TOTFBE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBE' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFBE , conn)
        df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO1' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFBO1 , conn)
        df_TOTFBO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO2' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE" + " AND " + "ModifyTime>=" + today_date_TOTFBO2 , conn)
    else:
        df_CASE = pd.read_sql("SELECT * FROM " + "[" + 'CASE' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_CRLF = pd.read_sql("SELECT * FROM " + "[" + 'CRLF' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE", conn)     
        # df_CRSF = pd.read_sql("SELECT * FROM " + "[" + 'CRSF' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_DEATH = pd.read_sql("SELECT * FROM " + "[" + 'DEATH' + "]" + "where ID=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        # df_LABD1 = pd.read_sql("SELECT * FROM " + "[" + 'LABD1' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        # df_LABD2 = pd.read_sql("SELECT * FROM " + "[" + 'LABD2' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_LABM1 = pd.read_sql("SELECT * FROM " + "[" + 'LABM1' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_LABM2 = pd.read_sql("SELECT * FROM " + "[" + 'LABM2' + "]" + "where h9=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFAE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAE' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO1' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFAO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO2' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFBE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBE' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO1' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)
        df_TOTFBO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO2' + "]" + "where d3=="+"'"+ID+"'" + "COLLATE NOCASE", conn)

    # df_CASE['ModifyTime'] = df_CASE['ModifyTime'].astype('datetime64[ns]')
    # df_CRLF['ModifyTime'] = df_CRLF['ModifyTime'].astype('datetime64[ns]')
    # df_CRSF['ModifyTime'] = df_CRSF['ModifyTime'].astype('datetime64[ns]')
    # df_DEATH['ModifyTime'] = df_DEATH['ModifyTime'].astype('datetime64[ns]')
    # df_LABD1['ModifyTime'] = df_LABD1['ModifyTime'].astype('datetime64[ns]')
    # df_LABD2['ModifyTime'] = df_LABD2['ModifyTime'].astype('datetime64[ns]')
    # df_LABM1['ModifyTime'] = df_LABM1['ModifyTime'].astype('datetime64[ns]')
    # df_LABM2['ModifyTime'] = df_LABM2['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFAE['ModifyTime'] = df_TOTFAE['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFAO1['ModifyTime'] = df_TOTFAO1['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFAO2['ModifyTime'] = df_TOTFAO2['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFBE['ModifyTime'] = df_TOTFBE['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFBO1['ModifyTime'] = df_TOTFBO1['ModifyTime'].astype('datetime64[ns]')
    # df_TOTFBO2['ModifyTime'] = df_TOTFBO2['ModifyTime'].astype('datetime64[ns]')
    
    # if stats == 'update':
    #     df_CASE = df_CASE.query("ModifyTime >=" + today_date)
    #     df_CRLF = df_CRLF.query("ModifyTime >=" + today_date) 
    #     df_CRSF = df_CRSF.query("ModifyTime >=" + today_date)
    #     df_DEATH = df_DEATH.query("ModifyTime >=" + today_date)
    #     df_LABD1 = df_LABD1.query("ModifyTime >=" + today_date)
    #     df_LABD2 = df_LABD2.query("ModifyTime >=" + today_date)
    #     df_LABM1 = df_LABM1.query("ModifyTime >=" + today_date)
    #     df_LABM2 = df_LABM2.query("ModifyTime >=" + today_date)
    #     df_TOTFAE = df_TOTFAE.query("ModifyTime >=" + today_date)
    #     df_TOTFAO1 = df_TOTFAO1.query("ModifyTime >=" + today_date)
    #     df_TOTFAO2 = df_TOTFAO2.query("ModifyTime >=" + today_date)
    #     df_TOTFBE = df_TOTFBE.query("ModifyTime >=" + today_date)
    #     df_TOTFBO1 = df_TOTFBO1.query("ModifyTime >=" + today_date)
    #     df_TOTFBO2 = df_TOTFBO2.query("ModifyTime >=" + today_date)
    # else:
    #     pass
    
    # df_LABD1.rename(columns={'h9': 'id'}, inplace=True)
    # df_LABD2.rename(columns={'h9': 'id'}, inplace=True)
    df_LABM1.rename(columns={'h9': 'id'}, inplace=True)
    df_LABM2.rename(columns={'h9': 'id'}, inplace=True)
    df_TOTFAE.rename(columns={'d3': 'id'}, inplace=True)
    df_TOTFAO1.rename(columns={'d3': 'id'}, inplace=True)
    df_TOTFAO2.rename(columns={'d3': 'id'}, inplace=True)
    df_TOTFBE.rename(columns={'d3': 'id'}, inplace=True)
    df_TOTFBO1.rename(columns={'d3': 'id'}, inplace=True)
    df_TOTFBO2.rename(columns={'d3': 'id'}, inplace=True)
    ############################################### TOTFB #############################################################

    ## BE join BO1
    global df_TOTFB, df_TOTFB_25_29, df_TOTFB_sult
    df_TOTFB = df_TOTFBE
    # df_TOTFB = pd.merge(df_TOTFBE, df_TOTFBO1, how='left', on=['id'], indicator=False, suffixes=('_BE', '_BO1'))
    df_TOTFB_25_29 = df_TOTFB[['id','d25','d26','d27','d28','d29']]

                                                ## CCI - onehot ##
    ## Myocardial infarction 心肌梗塞 (創與TOTFB合併的初表，取名為 df_TOTFB_sult，後繼續疊加其他疾病欄)
    _25 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d25'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _26 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d26'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _27 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d27'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _28 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d28'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _29 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d29'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _25_29 = pd.concat([_25,_26,_27,_28,_29]).drop_duplicates()
    _25_29['BO_Myocardial_infarction']='1'
    df_TOTFB_sult = pd.merge(df_TOTFB, _25_29, how='outer', on=['id'], indicator=False).fillna(value=0)

    def TOTFB_onehot(disease_Name, ICD):
        global df_TOTFB_sult,TOTFB_cci_onehot
        _25 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d25'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _26 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d26'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _27 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d27'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _28 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d28'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _29 = df_TOTFB_25_29.loc[df_TOTFB_25_29['d29'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _25_29 = pd.concat([_25,_26,_27,_28,_29]).drop_duplicates()
        _25_29[disease_Name]='1'
        df_TOTFB_sult = pd.merge(df_TOTFB_sult, _25_29, how='outer', on=['id'], indicator=False).fillna(value=0)
        return df_TOTFB_sult

    # ## Congestive heart failure 充血性心力衰竭
    df_TOTFB_sult = TOTFB_onehot('BO_Congestive_heart_failure',  ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'))

    # ## Peripheral vascular disease 周邊血管疾病
    df_TOTFB_sult = TOTFB_onehot('BO_Peripheral_vascular_disease', ('0930','4373','440','441','4431','4432','4438','4439','4471','5571','5579','V434','I70','I71','I731','I738','I739','I771','I790','I791','I798','K551','K558','K559','Z958','Z959'))

    # ## Cerebrovascular disease 腦血管疾病
    df_TOTFB_sult = TOTFB_onehot('BO_Cerebrovascular_disease', ('36234','430','431','432','433','434','435','436','437','438','G45','G46','H340','H341','H342','I60','I61','I62','I63','I64','I65','I66','I67','I68'))

    # ## Dementia 失智
    df_TOTFB_sult = TOTFB_onehot('BO_Dementia', ('2900','2901','2902','2903','2904','2940','2941','2942','2948','3310','3311','3312','3317','797','F01','F02','F03','F04','F05','F061','F068','G132','G138','G30','G310','G311','G312','G914','G94','R4181','R54'))

    # ## Chronic pulmonary disease 慢性肺病
    df_TOTFB_sult = TOTFB_onehot('BO_Chronic_pulmonary_disease', ('490','491','492','493','494','495','496','500','501','502','503','504','505','5064','5081','5088','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703'))

    # ## Rheumatic disease 風濕病
    df_TOTFB_sult = TOTFB_onehot('BO_Rheumatic_disease', ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148','725','M05','M06','M315','M32','M33','M34','M351','M353','M360'))

    # ## Peptic ulcer disease 消化性潰瘍病
    df_TOTFB_sult = TOTFB_onehot('BO_Peptic_ulcer_disease', ('531','532','533','534','K25','K26','K27','K28'))

    # ## Liver disease mild 肝病(輕度)
    df_TOTFB_sult = TOTFB_onehot('BO_Liver_disease_mild', ('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427','B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944'))

    # ## Diabetes without chronic complications 無慢性併發症的糖尿病
    df_TOTFB_sult = TOTFB_onehot('BO_Diabetes_without_chronic_complications', ('2508','2509','2490','2491','2492','2493','2499','E080','E090','E100','E110','E130','E081','E091','E101','E111','E131','E086','E096','E106','E116','E136','E088','E098','E108','E118','E138','E089','E099','E109','E119','E139'))

    # ## Renal disease, mild to moderate 腎病，輕度至中度
    df_TOTFB_sult = TOTFB_onehot('BO_Renal_disease_mild_to_moderate', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))

    # ## Diabetes with chronic complications 有慢性併發症的糖尿病
    df_TOTFB_sult = TOTFB_onehot('BO_Diabetes_with_chronic_complications', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))

    # ## Hemiplegia or paraplegia 偏癱或截癱
    df_TOTFB_sult = TOTFB_onehot('BO_Hemiplegia_or_paraplegia', ('3341','342','343','344','G041','G114','G800','G801','G802','G81','G82','G83'))

    # ## Any malignancy 任何惡性腫瘤
    df_TOTFB_sult = TOTFB_onehot('BO_Any_malignancy', ('14',' 15',' 16',' 170',' 171',' 172',' 174',' 175',' 176',' 179',' 18',' 190',' 191',' 192',' 193',' 194',' 195',' 1991',' 200',' 201',' 202',' 203',' 204',' 205',' 206',' 207',' 208',' 2386','C0','C1','C2','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','58','C60','C61','C62','C63','63','C76','C801','C81','C82','C83','C84','C85','C88','C9'))

    # ## Liver disease, moderate to severe 肝病，中度至重度
    df_TOTFB_sult = TOTFB_onehot('BO_Liver_disease_moderate_to_severe', ('4560','4561','4562x','5722','5723','5724','5728','I850','I864','K704','K711','K721','K729','K765','K766','K767'))

    # ## Renal disease, severe 腎病，嚴重
    df_TOTFB_sult = TOTFB_onehot('BO_Renal_disease_severe',('40301','40311','40391','40402','40403','40412','40413','40492','40493','5855','5856','586','5880','V4511','V4512','V560','V561','V562','V5631','V5632','V568','I120','I1311','I132','N185','N186','N19','N250','Z49','Z992'))

    # ## HIV infection, no AIDS 愛滋病毒感染，沒有愛滋病
    df_TOTFB_sult = TOTFB_onehot('BO_HIV_infection_no_AIDS', ('042','B20'))

    # ## Metastatic solid tumor 轉移性實體瘤
    df_TOTFB_sult = TOTFB_onehot('BO_Metastatic_solid_tumor', ('196','197','198','1990','C77','C78','C79','C800','C802'))

    # ## AIDS 愛滋病
    df_TOTFB_sult = TOTFB_onehot('BO_AIDS', ('112','180','114','1175','0074','0785','3483','054','115','0072','176','200','201','202','203','204','205','206','207','208','209','031','010','011','012','013','014','015','016','017','018','018','1363','V1261','0463','0031','130','7994','B37','C53','B38','B45','A072','B25','G934','B00','B39','A073','C46','C81','C82','C83','C84','C85','C86','C87','C88','C89','C90','C91','C92','C93','C94','C95','C96','C96','A31','A15','A16','A17','A18','A19','B59','Z8701','A812','A021','B58','R64'))

    # ## 口腔、口咽及下咽
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_lip_oral_cavity_and_pharynx', ('C00','C01','C02','C03','C04','C05','C06','C09','C10','C12','C13','C14','140','141','143','144','145','146','148','149'))

    # ## 口腔
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_oral_cavity', ('C00','C01','C02','C03','C04','C05','C06','140','141','143','144','145'))

    # ## 唇
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_lip', ('C00'))

    # ## 舌
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_tongue', ('C01','C02'))

    # ## 齒齦
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_gum', ('C03'))

    # ## 口底
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_floor_of_mouth', ('C04'))

    # ## 口腔之其他及未詳細說明部位
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_other_and_unspecified_parts_of_mouth', ('C05','C06'))

    # ## 口咽
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_oropharynx', ('C09','C10'))

    # ## 下咽
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_hypopharynx', ('C12','C13'))

    # ## 咽和唇、口腔及咽之分界不明部位
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx', ('C14'))

    # ## 主唾液腺
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_major_salivary_glands', ('C07','C08','142'))

    # ## 鼻咽
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_nasopharynx', ('C11','147'))

    # ## 消化器官及腹膜
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_digestive_organs_and_peritoneum', ('C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C48'))

    # ## 食道
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_esophagus', ('C15','150'))

    # ## 胃
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_stomach', ('C16','151'))

    # ## 小腸
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_small_intestine', ('C17','152'))

    # ## 結腸、直腸、乙狀結腸連結部及肛門
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_colon_rectum_rectosigmoid_junction_and_anus', ('C18','C19','C20','C21','153','154'))

    # ## 結腸
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_colon', ('C18','153'))

    # ## 直腸、乙狀結腸連結部及肛門
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_rectum_rectosigmoid_junction_and_anus', ('C19','C20','C21','154'))

    # ## 肝及肝內膽管
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_liver_and_intrahepatic_bile_ducts', ('C22','155'))

    # ## 膽囊及肝外膽管
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_gallbladder_and_extrahepatic_bile_ducts', ('C23','C24','156'))

    # ## 胰
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_pancreas', ('C25','157'))

    # ## 後腹膜腔及腹膜
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_retroperitoneum_and_peritoneum', ('C48','158'))

    # ## 消化器官之其他部位及與腹膜分界不明部位
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum', ('C26','159'))

    # ## 呼吸系統及胸腔內器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_respiratory_system_and_intrathoracic_organs', ('C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','160','161','162','163','164'))

    # ## 鼻腔、副竇、中耳及內耳
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear', ('C30','C31','160'))

    # ## 喉
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_larynx', ('C32','161'))

    # ## 肺、支氣管及氣管
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_trachea_bronchus_and_lung', ('C33','C34','162'))

    # ## 胸膜
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_pleura', ('C384','163'))

    # ## 胸腺、心臟及縱膈
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_thymus_heart_and_mediastinum', ('C37','C380','C381','C382','C383','C388','164'))

    # ## 骨、關節及關節軟骨
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_bones_joints_and_articular_cartilage', ('C40','C41','170'))

    # ## 結締組織、軟組織及其他皮下組織
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_connective_subcutaneous_and_other_soft_tissues', ('C47','C49','171'))

    # ## 皮膚
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_skin', ('C43','C44','172','173'))

    # ## 乳房
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_breast', ('C50','174','175','C50'))

    # ## 女性生殖器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_female_genital_organs', ('C51','C52','C53','C54','C55','C56','C57','C58'))

    # ## 子宮頸
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_cervix_uteri', ('C53','C55','179','180'))

    # ## 子宮體
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_corpus_uteri', ('C54','182'))

    # ## 卵巢、輸卵管及寬韌帶
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_ovary_fallopian_tube_and_broad_ligament', ('C56','C570','C571','C572','C573','C574','183'))

    # ## 陰道、外陰部及其他未詳細說明之女性生殖器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_vagina_vulva_other_and_unspecified_female_genital_organs', ('C51','C52','C577','C578','C579','184'))

    # ## 男性生殖器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_male_genital_organs', ('C60','C61','C62','C63'))

    # ## 攝護腺
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_prostate_gland', ('C61','185'))

    # ## 睪丸
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_testis', ('C62','186'))

    # ## 陰莖及其他男性生殖器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_penis_and_other_male_genital_organs', ('C60','C61','C62','C63','187'))

    # ## 泌尿器官
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_urinary_tract', ('C64','C65','C66','C67','C68'))

    # ## 膀胱
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_bladder', ('C67','188'))

    # ## 腎
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_kidney', ('C64','1890'))

    # ## 腎盂及其他泌尿系統
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_renal_pelvis', ('C65','C66','C68','1891','1892','1893','1894','1895','1896','1897','1898','1899'))

    # ## 眼及淚腺
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_eye_and_lacrimal_gland', ('C69','190'))

    # ## 中樞神經系統
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_brain_and_other_parts_of_central_nervous_system', ('C70','C71','C72','191','192'))

    # ## 腦
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_brain', ('C71','191'))

    # ## 腦膜、脊髓及神經系統未詳細說明部位
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system', ('C70','C72','192'))

    # ## 甲狀腺
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_thyroid_gland', ('C73','193'))

    # ## 其他內分泌腺
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_other_endocrine_glands', ('C74','C75','194'))

    # ## 不明原發部位
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_unknown_primary_sites', ('C80','199'))

    # ## 白血病
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_leukemia', ('M980','M981','M982','M983','M984','M985','M986','M987','M988','M989','M990','M991','M992','M993','M994','M995','M996','M997','M998','M999','204','205','206','207','208','C91','C92','C93','C94','C95'))

    # ## 惡性淋巴瘤
    df_TOTFB_sult = TOTFB_onehot('BO_Mal_malignant_lymphoma', ('M959','M960','M961','M962','M963','M964','M965','M966','M967','M968','M969','M970','M971','M972','M973','M974','M975','M976'))

    # ## 糖尿病Cohort
    df_TOTFB_sult = TOTFB_onehot('BO_Diabetes', ('E08','E09','E10','E11','E12','E13','250'))

    # ## 阿茲海默氏病Cohort
    df_TOTFB_sult = TOTFB_onehot('BO_Alzheimer', ('G30','3310'))

    # ## 慢性腎臟病Cohort
    df_TOTFB_sult = TOTFB_onehot('BO_CKD', ('A1811','A5275','C64','C65','C689','D300','D3A093','D410','D411','D422','D593','E082','E092','E1021','E1022','E1029','E1065','E112','E1165','E132','E260','E261','E268','E269','E7203','E723','E728','E740','E744','E748','E7521','E7522','E7524','E753','E77','I12','I13','I701','I722','I7581','I773','I7773','K767','M103','M1A10','M1A10','M1A1110','M1A1111','M1A1120','M1A1121','M1A1190','M1A1191','M1A1210','M1A1211','M1A1220','M1A1221','M1A1290','M1A1291','M1A1310','M1A1311','M1A1320','M1A1321','M1A1390','M1A1391','M1A1410','M1A1411','M1A1420','M1A1421','M1A1490','M1A1491','M1A1510','M1A1511','M1A1520','M1A1521','M1A1590','M1A1591','M1A1610','M1A1611','M1A1620','M1A1621','M1A1690','M1A1691','M1A1710','M1A1711','M1A1720','M1A1721','M1A1790','M1A1791','M1A18','M1A18','M1A19','M1A19','M3214','M3215','M3504','N00','N008','N009','N01','N02','N03','N04','N05','N06','N07','N08','N131','N132','N133','N135','N139','N14','N150','N158','N159','N16','N17','N18','N19','N200','N25','N261','N269','N271','N279','N289','N29','O104','O121','O2683','Q60','Q6102','Q611','Q612','Q613','Q614','Q615','Q618','Q620','Q621','Q622','Q6231','Q6232','Q6239','Q63','Q851','R944','T560','T560','T560','T560','Z4822','Z524','Z940','Z992','01600','01601','01602','01603','01604','01605','01606','0954','1890','1891','1899','2230','23691','25040','25041','25042','25043','2551','2708','2710','2714','2727','27410','27411','27419','28311','40300','40301','40310','40311','40390','40391','40400','40401','40402','40403','40410','40411','40412','40413','40490','40491','40492','40493','4401','4421','4473','5724','5800','5804','58081','58089','5809','5810','5811','5812','5813','58181','58189','5819','5820','5821','5822','5823','5824','58281','58289','5829','5830','5831','5832','5834','5836','5837','58381','58389','5839','5845','5846','5847','5848','5849','585','586','587','5880','5881','5888','5889','5891','5899','591','5933','5939','5996','64210','64211','64212','64213','64214','64620','64621','64622','64623','64624','7530','75312','75313','75314','75315','75316','75317','75319','75320','75321','75322','75329','7533','7595','7944','9849','V420','V451','V594'))


    #取唯一筆
    df_TOTFB_sult_final = df_TOTFB_sult[['id','BO_Myocardial_infarction','BO_Mal_larynx','BO_Congestive_heart_failure','BO_Peripheral_vascular_disease','BO_Cerebrovascular_disease','BO_Dementia','BO_Chronic_pulmonary_disease','BO_Rheumatic_disease','BO_Peptic_ulcer_disease','BO_Liver_disease_mild','BO_Diabetes_without_chronic_complications','BO_Renal_disease_mild_to_moderate','BO_Diabetes_with_chronic_complications','BO_Hemiplegia_or_paraplegia','BO_Any_malignancy','BO_Liver_disease_moderate_to_severe','BO_Renal_disease_severe','BO_HIV_infection_no_AIDS','BO_Metastatic_solid_tumor','BO_AIDS','BO_Mal_lip_oral_cavity_and_pharynx','BO_Mal_oral_cavity','BO_Mal_lip','BO_Mal_tongue','BO_Mal_gum','BO_Mal_floor_of_mouth','BO_Mal_other_and_unspecified_parts_of_mouth','BO_Mal_oropharynx','BO_Mal_hypopharynx','BO_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx','BO_Mal_major_salivary_glands','BO_Mal_nasopharynx','BO_Mal_digestive_organs_and_peritoneum','BO_Mal_esophagus','BO_Mal_stomach','BO_Mal_small_intestine','BO_Mal_colon_rectum_rectosigmoid_junction_and_anus','BO_Mal_colon','BO_Mal_rectum_rectosigmoid_junction_and_anus','BO_Mal_liver_and_intrahepatic_bile_ducts','BO_Mal_gallbladder_and_extrahepatic_bile_ducts','BO_Mal_pancreas','BO_Mal_retroperitoneum_and_peritoneum','BO_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum','BO_Mal_respiratory_system_and_intrathoracic_organs','BO_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear','BO_Mal_trachea_bronchus_and_lung','BO_Mal_pleura','BO_Mal_thymus_heart_and_mediastinum','BO_Mal_bones_joints_and_articular_cartilage','BO_Mal_connective_subcutaneous_and_other_soft_tissues','BO_Mal_skin','BO_Mal_breast','BO_Mal_female_genital_organs','BO_Mal_cervix_uteri','BO_Mal_corpus_uteri','BO_Mal_ovary_fallopian_tube_and_broad_ligament','BO_Mal_vagina_vulva_other_and_unspecified_female_genital_organs','BO_Mal_male_genital_organs','BO_Mal_prostate_gland','BO_Mal_testis','BO_Mal_penis_and_other_male_genital_organs','BO_Mal_urinary_tract','BO_Mal_bladder','BO_Mal_kidney','BO_Mal_renal_pelvis','BO_Mal_eye_and_lacrimal_gland','BO_Mal_brain_and_other_parts_of_central_nervous_system','BO_Mal_brain','BO_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system','BO_Mal_thyroid_gland','BO_Mal_other_endocrine_glands','BO_Mal_unknown_primary_sites','BO_Mal_leukemia','BO_Mal_malignant_lymphoma','BO_Diabetes','BO_Alzheimer','BO_CKD']]
    df_TOTFB_sult_final = df_TOTFB_sult_final.drop_duplicates(subset = 'id')
    TOTFB_cci_onehot = df_TOTFB_sult_final.copy()

    ############################################### TOTFB #############################################################

                                            ## CCI - count  ##
    def TOTFB_count(disease_Name, ICD):
        global r_e_combine,TOTFB_cci_count,TOTFB_order_code
        _25 = df_TOTFBE.loc[df_TOTFBE['d25'].str.startswith((ICD), na = False)]
        _26 = df_TOTFBE.loc[df_TOTFBE['d26'].str.startswith((ICD), na = False)]
        _27 = df_TOTFBE.loc[df_TOTFBE['d27'].str.startswith((ICD), na = False)]
        _28 = df_TOTFBE.loc[df_TOTFBE['d28'].str.startswith((ICD), na = False)]
        _29 = df_TOTFBE.loc[df_TOTFBE['d29'].str.startswith((ICD), na = False)]
        
        col_combine = pd.concat([_25,_26,_27,_28,_29])
        
        early_time = col_combine.drop_duplicates(subset = 'id')
        early_time = early_time[['id','t2','d10']]
        early_time.rename(columns={'t2': disease_Name+'_t2','d10':disease_Name + '_d10'}, inplace=True)
        
        col_combine = col_combine.drop_duplicates(subset = ['verify'])
        row_count = col_combine['id'].value_counts() 
        row_count = row_count.to_frame()
        row_count = row_count.reset_index()
        row_count.rename(columns={'id': disease_Name,'index':'id'}, inplace=True)
        
        r_e_combine = pd.merge(row_count, early_time, how='outer', on=['id'], indicator=False).fillna(value=0)
        return r_e_combine

    df_TOTFB_sult_final2_0 = TOTFB_count('BC_Myocardial_infarction', ('410','412','I21','I22','I252'))
    df_TOTFB_sult_final2_1 = TOTFB_count('BC_Congestive_heart_failure', ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'))
    df_TOTFB_sult_final2_2 = TOTFB_count('BC_Peripheral_vascular_disease', ('0930','4373','440','441','4431','4432','4438','4439','4471','5571','5579','V434','I70','I71','I731','I738','I739','I771','I790','I791','I798','K551','K558','K559','Z958','Z959'))
    df_TOTFB_sult_final2_3 = TOTFB_count('BC_Cerebrovascular_disease', ('36234','430','431','432','433','434','435','436','437','438','G45','G46','H340','H341','H342','I60','I61','I62','I63','I64','I65','I66','I67','I68'))
    df_TOTFB_sult_final2_4 = TOTFB_count('BC_Dementia', ('2900','2901','2902','2903','2904','2940','2941','2942','2948','3310','3311','3312','3317','797','F01','F02','F03','F04','F05','F061','F068','G132','G138','G30','G310','G311','G312','G914','G94','R4181','R54'))
    df_TOTFB_sult_final2_5 = TOTFB_count('BC_Chronic_pulmonary_disease', ('490','491','492','493','494','495','496','500','501','502','503','504','505','5064','5081','5088','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703'))
    df_TOTFB_sult_final2_6 = TOTFB_count('BC_Rheumatic_disease', ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148','725','M05','M06','M315','M32','M33','M34','M351','M353','M360'))
    df_TOTFB_sult_final2_7 = TOTFB_count('BC_Peptic_ulcer_disease', ('531','532','533','534','K25','K26','K27','K28'))
    df_TOTFB_sult_final2_8 = TOTFB_count('BC_Liver_disease_mild', ('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427','B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944'))
    df_TOTFB_sult_final2_9 = TOTFB_count('BC_Diabetes_without_chronic_complications', ('2508','2509','2490','2491','2492','2493','2499','E080','E090','E100','E110','E130','E081','E091','E101','E111','E131','E086','E096','E106','E116','E136','E088','E098','E108','E118','E138','E089','E099','E109','E119','E139'))
    df_TOTFB_sult_final2_10 = TOTFB_count('BC_Renal_disease_mild_to_moderate', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))
    df_TOTFB_sult_final2_11 = TOTFB_count('BC_Diabetes_with_chronic_complications', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))
    df_TOTFB_sult_final2_12 = TOTFB_count('BC_Hemiplegia_or_paraplegia', ('3341','342','343','344','G041','G114','G800','G801','G802','G81','G82','G83'))
    df_TOTFB_sult_final2_13 = TOTFB_count('BC_Any_malignancy', ('14',' 15',' 16',' 170',' 171',' 172',' 174',' 175',' 176',' 179',' 18',' 190',' 191',' 192',' 193',' 194',' 195',' 1991',' 200',' 201',' 202',' 203',' 204',' 205',' 206',' 207',' 208',' 2386','C0','C1','C2','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','58','C60','C61','C62','C63','63','C76','C801','C81','C82','C83','C84','C85','C88','C9'))
    df_TOTFB_sult_final2_14 = TOTFB_count('BC_Liver_disease_moderate_to_severe', ('4560','4561','4562x','5722','5723','5724','5728','I850','I864','K704','K711','K721','K729','K765','K766','K767'))
    df_TOTFB_sult_final2_15 = TOTFB_count('BC_Renal_disease_severe', ('40301','40311','40391','40402','40403','40412','40413','40492','40493','5855','5856','586','5880','V4511','V4512','V560','V561','V562','V5631','V5632','V568','I120','I1311','I132','N185','N186','N19','N250','Z49','Z992'))
    df_TOTFB_sult_final2_16 = TOTFB_count('BC_HIV_infection_no_AIDS', ('042','B20'))
    df_TOTFB_sult_final2_17 = TOTFB_count('BC_Metastatic_solid_tumor', ('196','197','198','1990','C77','C78','C79','C800','C802'))
    df_TOTFB_sult_final2_18 = TOTFB_count('BC_AIDS', ('112','180','114','1175','0074','0785','3483','054','115','0072','176','200','201','202','203','204','205','206','207','208','209','031','010','011','012','013','014','015','016','017','018','018','1363','V1261','0463','0031','130','7994','B37','C53','B38','B45','A072','B25','G934','B00','B39','A073','C46','C81','C82','C83','C84','C85','C86','C87','C88','C89','C90','C91','C92','C93','C94','C95','C96','C96','A31','A15','A16','A17','A18','A19','B59','Z8701','A812','A021','B58','R64'))
    df_TOTFB_sult_final2_19 = TOTFB_count('BC_Mal_lip_oral_cavity_and_pharynx', ('C00','C01','C02','C03','C04','C05','C06','C09','C10','C12','C13','C14','140','141','143','144','145','146','148','149'))
    df_TOTFB_sult_final2_20 = TOTFB_count('BC_Mal_oral_cavity', ('C00','C01','C02','C03','C04','C05','C06','140','141','143','144','145'))
    df_TOTFB_sult_final2_21 = TOTFB_count('BC_Mal_lip', ('C00'))
    df_TOTFB_sult_final2_22 = TOTFB_count('BC_Mal_tongue', ('C01','C02'))
    df_TOTFB_sult_final2_23 = TOTFB_count('BC_Mal_gum', ('C03'))
    df_TOTFB_sult_final2_24 = TOTFB_count('BC_Mal_floor_of_mouth', ('C04'))
    df_TOTFB_sult_final2_25 = TOTFB_count('BC_Mal_other_and_unspecified_parts_of_mouth', ('C05','C06'))
    df_TOTFB_sult_final2_26 = TOTFB_count('BC_Mal_oropharynx', ('C09','C10'))
    df_TOTFB_sult_final2_27 = TOTFB_count('BC_Mal_hypopharynx', ('C12','C13'))
    df_TOTFB_sult_final2_28 = TOTFB_count('BC_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx', ('C14'))
    df_TOTFB_sult_final2_29 = TOTFB_count('BC_Mal_major_salivary_glands', ('C07','C08','142'))
    df_TOTFB_sult_final2_30 = TOTFB_count('BC_Mal_nasopharynx', ('C11','147'))
    df_TOTFB_sult_final2_31 = TOTFB_count('BC_Mal_digestive_organs_and_peritoneum', ('C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C48'))
    df_TOTFB_sult_final2_32 = TOTFB_count('BC_Mal_esophagus', ('C15','150'))
    df_TOTFB_sult_final2_33 = TOTFB_count('BC_Mal_stomach', ('C16','151'))
    df_TOTFB_sult_final2_34 = TOTFB_count('BC_Mal_small_intestine', ('C17','152'))
    df_TOTFB_sult_final2_35 = TOTFB_count('BC_Mal_colon_rectum_rectosigmoid_junction_and_anus', ('C18','C19','C20','C21','153','154'))
    df_TOTFB_sult_final2_36 = TOTFB_count('BC_Mal_colon', ('C18','153'))
    df_TOTFB_sult_final2_37 = TOTFB_count('BC_Mal_rectum_rectosigmoid_junction_and_anus', ('C19','C20','C21','154'))
    df_TOTFB_sult_final2_38 = TOTFB_count('BC_Mal_liver_and_intrahepatic_bile_ducts', ('C22','155'))
    df_TOTFB_sult_final2_39 = TOTFB_count('BC_Mal_gallbladder_and_extrahepatic_bile_ducts', ('C23','C24','156'))
    df_TOTFB_sult_final2_40 = TOTFB_count('BC_Mal_pancreas', ('C25','157'))
    df_TOTFB_sult_final2_41 = TOTFB_count('BC_Mal_retroperitoneum_and_peritoneum', ('C48','158'))
    df_TOTFB_sult_final2_42 = TOTFB_count('BC_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum', ('C26','159'))
    df_TOTFB_sult_final2_43 = TOTFB_count('BC_Mal_respiratory_system_and_intrathoracic_organs', ('C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','160','161','162','163','164'))
    df_TOTFB_sult_final2_44 = TOTFB_count('BC_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear', ('C30','C31','160'))
    df_TOTFB_sult_final2_45 = TOTFB_count('BC_Mal_larynx', ('C32','161'))
    df_TOTFB_sult_final2_46 = TOTFB_count('BC_Mal_trachea_bronchus_and_lung', ('C33','C34','162'))
    df_TOTFB_sult_final2_47 = TOTFB_count('BC_Mal_pleura', ('C384','163'))
    df_TOTFB_sult_final2_48 = TOTFB_count('BC_Mal_thymus_heart_and_mediastinum', ('C37','C380','C381','C382','C383','C388','164'))
    df_TOTFB_sult_final2_49 = TOTFB_count('BC_Mal_bones_joints_and_articular_cartilage', ('C40','C41','170'))
    df_TOTFB_sult_final2_50 = TOTFB_count('BC_Mal_connective_subcutaneous_and_other_soft_tissues', ('C47','C49','171'))
    df_TOTFB_sult_final2_51 = TOTFB_count('BC_Mal_skin', ('C43','C44','172','173'))
    df_TOTFB_sult_final2_52 = TOTFB_count('BC_Mal_breast', ('C50','174','175','C50'))
    df_TOTFB_sult_final2_53 = TOTFB_count('BC_Mal_female_genital_organs', ('C51','C52','C53','C54','C55','C56','C57','C58'))
    df_TOTFB_sult_final2_54 = TOTFB_count('BC_Mal_cervix_uteri', ('C53','C55','179','180'))
    df_TOTFB_sult_final2_55 = TOTFB_count('BC_Mal_corpus_uteri', ('C54','182'))
    df_TOTFB_sult_final2_56 = TOTFB_count('BC_Mal_ovary_fallopian_tube_and_broad_ligament', ('C56','C570','C571','C572','C573','C574','183'))
    df_TOTFB_sult_final2_57 = TOTFB_count('BC_Mal_vagina_vulva_other_and_unspecified_female_genital_organs', ('C51','C52','C577','C578','C579','184'))
    df_TOTFB_sult_final2_58 = TOTFB_count('BC_Mal_male_genital_organs', ('C60','C61','C62','C63'))
    df_TOTFB_sult_final2_59 = TOTFB_count('BC_Mal_prostate_gland', ('C61','185'))
    df_TOTFB_sult_final2_60 = TOTFB_count('BC_Mal_testis', ('C62','186'))
    df_TOTFB_sult_final2_61 = TOTFB_count('BC_Mal_penis_and_other_male_genital_organs', ('C60','C61','C62','C63','187'))
    df_TOTFB_sult_final2_62 = TOTFB_count('BC_Mal_urinary_tract', ('C64','C65','C66','C67','C68'))
    df_TOTFB_sult_final2_63 = TOTFB_count('BC_Mal_bladder', ('C67','188'))
    df_TOTFB_sult_final2_64 = TOTFB_count('BC_Mal_kidney', ('C64','1890'))
    df_TOTFB_sult_final2_65 = TOTFB_count('BC_Mal_renal_pelvis', ('C65','C66','C68','1891','1892','1893','1894','1895','1896','1897','1898','1899'))
    df_TOTFB_sult_final2_66 = TOTFB_count('BC_Mal_eye_and_lacrimal_gland', ('C69','190'))
    df_TOTFB_sult_final2_67 = TOTFB_count('BC_Mal_brain_and_other_parts_of_central_nervous_system', ('C70','C71','C72','191','192'))
    df_TOTFB_sult_final2_68 = TOTFB_count('BC_Mal_brain', ('C71','191'))
    df_TOTFB_sult_final2_69 = TOTFB_count('BC_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system', ('C70','C72','192'))
    df_TOTFB_sult_final2_70 = TOTFB_count('BC_Mal_thyroid_gland', ('C73','193'))
    df_TOTFB_sult_final2_71 = TOTFB_count('BC_Mal_other_endocrine_glands', ('C74','C75','194'))
    df_TOTFB_sult_final2_72 = TOTFB_count('BC_Mal_unknown_primary_sites', ('C80','199'))
    df_TOTFB_sult_final2_73 = TOTFB_count('BC_Mal_leukemia', ('M980','M981','M982','M983','M984','M985','M986','M987','M988','M989','M990','M991','M992','M993','M994','M995','M996','M997','M998','M999','204','205','206','207','208','C91','C92','C93','C94','C95'))
    df_TOTFB_sult_final2_74 = TOTFB_count('BC_Mal_malignant_lymphoma', ('M959','M960','M961','M962','M963','M964','M965','M966','M967','M968','M969','M970','M971','M972','M973','M974','M975','M976'))
    df_TOTFB_sult_final2_75 = TOTFB_count('BC_Diabetes', ('E08','E09','E10','E11','E12','E13','250'))
    df_TOTFB_sult_final2_76 = TOTFB_count('BC_Alzheimer', ('G30','3310'))
    df_TOTFB_sult_final2_77 = TOTFB_count('BC_CKD', ('A1811','A5275','C64','C65','C689','D300','D3A093','D410','D411','D422','D593','E082','E092','E1021','E1022','E1029','E1065','E112','E1165','E132','E260','E261','E268','E269','E7203','E723','E728','E740','E744','E748','E7521','E7522','E7524','E753','E77','I12','I13','I701','I722','I7581','I773','I7773','K767','M103','M1A10','M1A10','M1A1110','M1A1111','M1A1120','M1A1121','M1A1190','M1A1191','M1A1210','M1A1211','M1A1220','M1A1221','M1A1290','M1A1291','M1A1310','M1A1311','M1A1320','M1A1321','M1A1390','M1A1391','M1A1410','M1A1411','M1A1420','M1A1421','M1A1490','M1A1491','M1A1510','M1A1511','M1A1520','M1A1521','M1A1590','M1A1591','M1A1610','M1A1611','M1A1620','M1A1621','M1A1690','M1A1691','M1A1710','M1A1711','M1A1720','M1A1721','M1A1790','M1A1791','M1A18','M1A18','M1A19','M1A19','M3214','M3215','M3504','N00','N008','N009','N01','N02','N03','N04','N05','N06','N07','N08','N131','N132','N133','N135','N139','N14','N150','N158','N159','N16','N17','N18','N19','N200','N25','N261','N269','N271','N279','N289','N29','O104','O121','O2683','Q60','Q6102','Q611','Q612','Q613','Q614','Q615','Q618','Q620','Q621','Q622','Q6231','Q6232','Q6239','Q63','Q851','R944','T560','T560','T560','T560','Z4822','Z524','Z940','Z992','01600','01601','01602','01603','01604','01605','01606','0954','1890','1891','1899','2230','23691','25040','25041','25042','25043','2551','2708','2710','2714','2727','27410','27411','27419','28311','40300','40301','40310','40311','40390','40391','40400','40401','40402','40403','40410','40411','40412','40413','40490','40491','40492','40493','4401','4421','4473','5724','5800','5804','58081','58089','5809','5810','5811','5812','5813','58181','58189','5819','5820','5821','5822','5823','5824','58281','58289','5829','5830','5831','5832','5834','5836','5837','58381','58389','5839','5845','5846','5847','5848','5849','585','586','587','5880','5881','5888','5889','5891','5899','591','5933','5939','5996','64210','64211','64212','64213','64214','64620','64621','64622','64623','64624','7530','75312','75313','75314','75315','75316','75317','75319','75320','75321','75322','75329','7533','7595','7944','9849','V420','V451','V594'))

    list_cci = [df_TOTFB_sult_final2_0 ,df_TOTFB_sult_final2_1 ,df_TOTFB_sult_final2_2 ,df_TOTFB_sult_final2_3 ,df_TOTFB_sult_final2_4 ,df_TOTFB_sult_final2_5 ,df_TOTFB_sult_final2_6 ,df_TOTFB_sult_final2_7 ,df_TOTFB_sult_final2_8 ,df_TOTFB_sult_final2_9, df_TOTFB_sult_final2_10, df_TOTFB_sult_final2_11, df_TOTFB_sult_final2_12, df_TOTFB_sult_final2_13, df_TOTFB_sult_final2_14, df_TOTFB_sult_final2_15, df_TOTFB_sult_final2_16, df_TOTFB_sult_final2_17, df_TOTFB_sult_final2_18, df_TOTFB_sult_final2_19, df_TOTFB_sult_final2_20, df_TOTFB_sult_final2_21, df_TOTFB_sult_final2_22, df_TOTFB_sult_final2_23, df_TOTFB_sult_final2_24, df_TOTFB_sult_final2_25, df_TOTFB_sult_final2_26, df_TOTFB_sult_final2_27, df_TOTFB_sult_final2_28, df_TOTFB_sult_final2_29, df_TOTFB_sult_final2_30, df_TOTFB_sult_final2_31, df_TOTFB_sult_final2_32, df_TOTFB_sult_final2_33, df_TOTFB_sult_final2_34, df_TOTFB_sult_final2_35, df_TOTFB_sult_final2_36, df_TOTFB_sult_final2_37, df_TOTFB_sult_final2_38, df_TOTFB_sult_final2_39, df_TOTFB_sult_final2_40, df_TOTFB_sult_final2_41, df_TOTFB_sult_final2_42, df_TOTFB_sult_final2_43, df_TOTFB_sult_final2_44, df_TOTFB_sult_final2_45, df_TOTFB_sult_final2_46, df_TOTFB_sult_final2_47, df_TOTFB_sult_final2_48, df_TOTFB_sult_final2_49, df_TOTFB_sult_final2_50, df_TOTFB_sult_final2_51, df_TOTFB_sult_final2_52, df_TOTFB_sult_final2_53, df_TOTFB_sult_final2_54, df_TOTFB_sult_final2_55, df_TOTFB_sult_final2_56, df_TOTFB_sult_final2_57, df_TOTFB_sult_final2_58, df_TOTFB_sult_final2_59, df_TOTFB_sult_final2_60, df_TOTFB_sult_final2_61, df_TOTFB_sult_final2_62, df_TOTFB_sult_final2_63, df_TOTFB_sult_final2_64, df_TOTFB_sult_final2_65, df_TOTFB_sult_final2_66, df_TOTFB_sult_final2_67, df_TOTFB_sult_final2_68, df_TOTFB_sult_final2_69, df_TOTFB_sult_final2_70, df_TOTFB_sult_final2_71, df_TOTFB_sult_final2_72, df_TOTFB_sult_final2_73, df_TOTFB_sult_final2_74, df_TOTFB_sult_final2_75, df_TOTFB_sult_final2_76, df_TOTFB_sult_final2_77]

    i=0
    for loop in range(len(list_cci)-1):
        if i<1:
            B_loop_table = pd.merge(list_cci[1], list_cci[0], how='outer', on=['id'], indicator=False).fillna(value=0)
            i+=2
            # print('if',i)
        else:
            B_loop_table = pd.merge(list_cci[i], B_loop_table, how='outer', on=['id'], indicator=False).fillna(value=0)
            i+=1
            # print('else',i)
            
    # B_loop_table = B_loop_table[['id','BC_Any_malignancy','BC_Any_malignancy_t2','BC_Any_malignancy_d10','BC_Metastatic_solid_tumor','BC_Metastatic_solid_tumor_t2','BC_Metastatic_solid_tumor_d10','BC_AIDS','BC_AIDS_t2','BC_AIDS_d10','BC_HIV_infection_no_AIDS','BC_HIV_infection_no_AIDS_t2','BC_HIV_infection_no_AIDS_d10','BC_Renal_disease_severe','BC_Renal_disease_severe_t2','BC_Renal_disease_severe_d10','BC_Liver_disease_moderate_to_severe','BC_Liver_disease_moderate_to_severe_t2','BC_Liver_disease_moderate_to_severe_d10','BC_Hemiplegia_or_paraplegia','BC_Hemiplegia_or_paraplegia_t2','BC_Hemiplegia_or_paraplegia_d10','BC_Renal_disease_mild_to_moderate','BC_Renal_disease_mild_to_moderate_t2','BC_Renal_disease_mild_to_moderate_d10','BC_Diabetes_with_chronic_complications','BC_Diabetes_with_chronic_complications_t2','BC_Diabetes_with_chronic_complications_d10','BC_Diabetes_without_chronic_complications','BC_Diabetes_without_chronic_complications_t2','BC_Diabetes_without_chronic_complications_d10','BC_Liver_disease_mild', 'BC_Liver_disease_mild_t2','BC_Liver_disease_mild_d10','BC_Peptic_ulcer_disease','BC_Peptic_ulcer_disease_t2','BC_Peptic_ulcer_disease_d10','BC_Rheumatic_disease','BC_Rheumatic_disease_t2','BC_Rheumatic_disease_d10','BC_Chronic_pulmonary_disease','BC_Chronic_pulmonary_disease_t2','BC_Chronic_pulmonary_disease_d10','BC_Dementia','BC_Dementia_t2','BC_Dementia_d10','BC_Cerebrovascular_disease','BC_Cerebrovascular_disease_t2','BC_Cerebrovascular_disease_d10','BC_Peripheral_vascular_disease','BC_Peripheral_vascular_disease_t2','BC_Peripheral_vascular_disease_d10','BC_Congestive_heart_failure','BC_Congestive_heart_failure_t2','BC_Congestive_heart_failure_d10','BC_Myocardial_infarction','BC_Myocardial_infarction_t2','BC_Myocardial_infarction_d10',
    #                              'BC_Mal_lip_oral_pharynx', 'BC_Mal_lip_oral_pharynx_t2', 'BC_Mal_lip_oral_pharynx_d10', 'BC_Mal_digestive_peritoneum', 'BC_Mal_digestive_peritoneum_t2', 'BC_Mal_digestive_peritoneum_d10', 'BC_Mal_respiratory_intrathoracic', 'BC_Mal_respiratory_intrathoracic_t2', 'BC_Mal_respiratory_intrathoracic_d10', 'BC_Mal_bone_articular', 'BC_Mal_bone_articular_t2', 'BC_Mal_bone_articular_d10', 'BC_Mal_skin', 'BC_Mal_skin_t2', 'BC_Mal_skin_d10', 'BC_Mal_mesothelial_soft_tissues', 'BC_Mal_mesothelial_soft_tissues_t2', 'BC_Mal_mesothelial_soft_tissues_d10', 'BC_Mal_breast', 'BC_Mal_breast_t2', 'BC_Mal_breast_d10', 'BC_Mal_female_reproductive', 'BC_Mal_female_reproductive_t2', 'BC_Mal_female_reproductive_d10', 'BC_Mal_male_reproductive', 'BC_Mal_male_reproductive_t2', 'BC_Mal_male_reproductive_d10', 'BC_Mal_urinary', 'BC_Mal_urinary_t2', 'BC_Mal_urinary_d10', 'BC_Mal_eye_brain_central_nervous', 'BC_Mal_eye_brain_central_nervous_t2', 'BC_Mal_eye_brain_central_nervous_d10', 'BC_Mal_thyroid_endocrine_glands', 'BC_Mal_thyroid_endocrine_glands_t2', 'BC_Mal_thyroid_endocrine_glands_d10', 'BC_Mal_Unspecified_sec', 'BC_Mal_Unspecified_sec_t2', 'BC_Mal_Unspecified_sec_d10', 'BC_Mal_lymphoid_hema_tissues', 'BC_Mal_lymphoid_hema_tissues_t2', 'BC_Mal_lymphoid_hema_tissues_d10', 'BC_Mal_Independent_mult', 'BC_Mal_Independent_mult_t2', 'BC_Mal_Independent_mult_d10', 'BC_Mal_Carcinoma_in_situ', 'BC_Mal_Carcinoma_in_situ_t2', 'BC_Mal_Carcinoma_in_situ_d10', 'BC_Mal_Benign_tumor', 'BC_Mal_Benign_tumor_t2', 'BC_Mal_Benign_tumor_d10', 'BC_Mal_unknown', 'BC_Mal_unknown_t2', 'BC_Mal_unknown_d10']]
    TOTFB_cci_count = B_loop_table.copy()

                                                ## B_order_code - count ##
    B_order_code = df_TOTFBO1[['id','p3']]
    total_counts = B_order_code.groupby(['id'])
    total_counts = total_counts.size().reset_index(name='B_p3_total_counts')

    B_order_code = df_TOTFBO1.drop_duplicates(subset = ['id','p3'])
    unique_counts = B_order_code.groupby(['id'])
    unique_counts = unique_counts.size().reset_index(name='B_p3_unique_counts')

    total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    TOTFB_order_code = total_unique_table.copy()

    ############################################### TOTFA #############################################################

    ## AE join BO1
    global df_TOTFA, df_TOTFA_19_23, df_TOTFA_sult
    df_TOTFA = df_TOTFAE
    # df_TOTFA = pd.merge(df_TOTFAE, df_TOTFAO1, how='left', on=['id'], indicator=False, suffixes=('_AE', '_AO1'))
    df_TOTFA_19_23 = df_TOTFA[['id','d19','d20','d21','d22','d23']]

                                                ## CCI - onehot ##
    ## Myocardial infarction 心肌梗塞 (創與TOTFA合併的初表，取名為 df_TOTFA_sult，後繼續疊加其他疾病欄)
    _19 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d19'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _20 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d20'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _21 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d21'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _22 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d22'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _23 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d23'].str.startswith(('410','412','I21','I22','I252'), na = False)][['id']].drop_duplicates()
    _19_23 = pd.concat([_19,_20,_21,_22,_23]).drop_duplicates()
    _19_23['AO_Myocardial_infarction']='1'
    df_TOTFA_sult = pd.merge(df_TOTFA, _19_23, how='outer', on=['id'], indicator=False).fillna(value=0)

    def TOTFA_onehot(disease_Name, ICD):
        global df_TOTFA_sult,TOTFA_cci_onehot
        _19 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d19'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _20 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d20'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _21 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d21'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _22 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d22'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _23 = df_TOTFA_19_23.loc[df_TOTFA_19_23['d23'].str.startswith((ICD), na = False)][['id']].drop_duplicates()
        _19_23 = pd.concat([_19,_20,_21,_22,_23]).drop_duplicates()
        _19_23[disease_Name]='1'
        df_TOTFA_sult = pd.merge(df_TOTFA_sult, _19_23, how='outer', on=['id'], indicator=False).fillna(value=0)
        return df_TOTFA_sult

    # ## Congestive heart failure 充血性心力衰竭
    df_TOTFA_sult = TOTFA_onehot('AO_Congestive_heart_failure',  ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'))

    # ## Peripheral vascular disease 周邊血管疾病
    df_TOTFA_sult = TOTFA_onehot('AO_Peripheral_vascular_disease', ('0930','4373','440','441','4431','4432','4438','4439','4471','5571','5579','V434','I70','I71','I731','I738','I739','I771','I790','I791','I798','K551','K558','K559','Z958','Z959'))

    # ## Cerebrovascular disease 腦血管疾病
    df_TOTFA_sult = TOTFA_onehot('AO_Cerebrovascular_disease', ('36234','430','431','432','433','434','435','436','437','438','G45','G46','H340','H341','H342','I60','I61','I62','I63','I64','I65','I66','I67','I68'))

    # ## Dementia 失智
    df_TOTFA_sult = TOTFA_onehot('AO_Dementia', ('2900','2901','2902','2903','2904','2940','2941','2942','2948','3310','3311','3312','3317','797','F01','F02','F03','F04','F05','F061','F068','G132','G138','G30','G310','G311','G312','G914','G94','R4181','R54'))

    # ## Chronic pulmonary disease 慢性肺病
    df_TOTFA_sult = TOTFA_onehot('AO_Chronic_pulmonary_disease', ('490','491','492','493','494','495','496','500','501','502','503','504','505','5064','5081','5088','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703'))

    # ## Rheumatic disease 風濕病
    df_TOTFA_sult = TOTFA_onehot('AO_Rheumatic_disease', ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148','725','M05','M06','M315','M32','M33','M34','M351','M353','M360'))

    # ## Peptic ulcer disease 消化性潰瘍病
    df_TOTFA_sult = TOTFA_onehot('AO_Peptic_ulcer_disease', ('531','532','533','534','K25','K26','K27','K28'))

    # ## Liver disease mild 肝病(輕度)
    df_TOTFA_sult = TOTFA_onehot('AO_Liver_disease_mild', ('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427','B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944'))

    # ## Diabetes without chronic complications 無慢性併發症的糖尿病
    df_TOTFA_sult = TOTFA_onehot('AO_Diabetes_without_chronic_complications', ('2508','2509','2490','2491','2492','2493','2499','E080','E090','E100','E110','E130','E081','E091','E101','E111','E131','E086','E096','E106','E116','E136','E088','E098','E108','E118','E138','E089','E099','E109','E119','E139'))

    # ## Renal disease, mild to moderate 腎病，輕度至中度
    df_TOTFA_sult = TOTFA_onehot('AO_Renal_disease_mild_to_moderate', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))

    # ## Diabetes with chronic complications 有慢性併發症的糖尿病
    df_TOTFA_sult = TOTFA_onehot('AO_Diabetes_with_chronic_complications', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))

    # ## Hemiplegia or paraplegia 偏癱或截癱
    df_TOTFA_sult = TOTFA_onehot('AO_Hemiplegia_or_paraplegia', ('3341','342','343','344','G041','G114','G800','G801','G802','G81','G82','G83'))

    # ## Any malignancy 任何惡性腫瘤
    df_TOTFA_sult = TOTFA_onehot('AO_Any_malignancy', ('14',' 15',' 16',' 170',' 171',' 172',' 174',' 175',' 176',' 179',' 18',' 190',' 191',' 192',' 193',' 194',' 195',' 1991',' 200',' 201',' 202',' 203',' 204',' 205',' 206',' 207',' 208',' 2386','C0','C1','C2','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','58','C60','C61','C62','C63','63','C76','C801','C81','C82','C83','C84','C85','C88','C9'))

    # ## Liver disease, moderate to severe 肝病，中度至重度
    df_TOTFA_sult = TOTFA_onehot('AO_Liver_disease_moderate_to_severe', ('4560','4561','4562x','5722','5723','5724','5728','I850','I864','K704','K711','K721','K729','K765','K766','K767'))

    # ## Renal disease, severe 腎病，嚴重
    df_TOTFA_sult = TOTFA_onehot('AO_Renal_disease_severe',('40301','40311','40391','40402','40403','40412','40413','40492','40493','5855','5856','586','5880','V4511','V4512','V560','V561','V562','V5631','V5632','V568','I120','I1311','I132','N185','N186','N19','N250','Z49','Z992'))

    # ## HIV infection, no AIDS 愛滋病毒感染，沒有愛滋病
    df_TOTFA_sult = TOTFA_onehot('AO_HIV_infection_no_AIDS', ('042','B20'))

    # ## Metastatic solid tumor 轉移性實體瘤
    df_TOTFA_sult = TOTFA_onehot('AO_Metastatic_solid_tumor', ('196','197','198','1990','C77','C78','C79','C800','C802'))

    # ## AIDS 愛滋病
    df_TOTFA_sult = TOTFA_onehot('AO_AIDS', ('112','180','114','1175','0074','0785','3483','054','115','0072','176','200','201','202','203','204','205','206','207','208','209','031','010','011','012','013','014','015','016','017','018','018','1363','V1261','0463','0031','130','7994','B37','C53','B38','B45','A072','B25','G934','B00','B39','A073','C46','C81','C82','C83','C84','C85','C86','C87','C88','C89','C90','C91','C92','C93','C94','C95','C96','C96','A31','A15','A16','A17','A18','A19','B59','Z8701','A812','A021','B58','R64'))

   # ## 口腔、口咽及下咽
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_lip_oral_cavity_and_pharynx', ('C00','C01','C02','C03','C04','C05','C06','C09','C10','C12','C13','C14','140','141','143','144','145','146','148','149'))

    # ## 口腔
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_oral_cavity', ('C00','C01','C02','C03','C04','C05','C06','140','141','143','144','145'))

    # ## 唇
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_lip', ('C00'))

    # ## 舌
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_tongue', ('C01','C02'))

    # ## 齒齦
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_gum', ('C03'))

    # ## 口底
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_floor_of_mouth', ('C04'))

    # ## 口腔之其他及未詳細說明部位
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_other_and_unspecified_parts_of_mouth', ('C05','C06'))

    # ## 口咽
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_oropharynx', ('C09','C10'))

    # ## 下咽
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_hypopharynx', ('C12','C13'))

    # ## 咽和唇、口腔及咽之分界不明部位
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx', ('C14'))

    # ## 主唾液腺
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_major_salivary_glands', ('C07','C08','142'))

    # ## 鼻咽
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_nasopharynx', ('C11','147'))

    # ## 消化器官及腹膜
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_digestive_organs_and_peritoneum', ('C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C48'))

    # ## 食道
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_esophagus', ('C15','150'))

    # ## 胃
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_stomach', ('C16','151'))

    # ## 小腸
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_small_intestine', ('C17','152'))

    # ## 結腸、直腸、乙狀結腸連結部及肛門
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_colon_rectum_rectosigmoid_junction_and_anus', ('C18','C19','C20','C21','153','154'))

    # ## 結腸
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_colon', ('C18','153'))

    # ## 直腸、乙狀結腸連結部及肛門
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_rectum_rectosigmoid_junction_and_anus', ('C19','C20','C21','154'))

    # ## 肝及肝內膽管
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_liver_and_intrahepatic_bile_ducts', ('C22','155'))

    # ## 膽囊及肝外膽管
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_gallbladder_and_extrahepatic_bile_ducts', ('C23','C24','156'))

    # ## 胰
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_pancreas', ('C25','157'))

    # ## 後腹膜腔及腹膜
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_retroperitoneum_and_peritoneum', ('C48','158'))

    # ## 消化器官之其他部位及與腹膜分界不明部位
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum', ('C26','159'))

    # ## 呼吸系統及胸腔內器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_respiratory_system_and_intrathoracic_organs', ('C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','160','161','162','163','164'))

    # ## 鼻腔、副竇、中耳及內耳
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear', ('C30','C31','160'))

    # ## 喉
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_larynx', ('C32','161'))

    # ## 肺、支氣管及氣管
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_trachea_bronchus_and_lung', ('C33','C34','162'))

    # ## 胸膜
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_pleura', ('C384','163'))

    # ## 胸腺、心臟及縱膈
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_thymus_heart_and_mediastinum', ('C37','C380','C381','C382','C383','C388','164'))

    # ## 骨、關節及關節軟骨
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_bones_joints_and_articular_cartilage', ('C40','C41','170'))

    # ## 結締組織、軟組織及其他皮下組織
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_connective_subcutaneous_and_other_soft_tissues', ('C47','C49','171'))

    # ## 皮膚
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_skin', ('C43','C44','172','173'))

    # ## 乳房
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_breast', ('C50','174','175','C50'))

    # ## 女性生殖器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_female_genital_organs', ('C51','C52','C53','C54','C55','C56','C57','C58'))

    # ## 子宮頸
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_cervix_uteri', ('C53','C55','179','180'))

    # ## 子宮體
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_corpus_uteri', ('C54','182'))

    # ## 卵巢、輸卵管及寬韌帶
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_ovary_fallopian_tube_and_broad_ligament', ('C56','C570','C571','C572','C573','C574','183'))

    # ## 陰道、外陰部及其他未詳細說明之女性生殖器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_vagina_vulva_other_and_unspecified_female_genital_organs', ('C51','C52','C577','C578','C579','184'))

    # ## 男性生殖器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_male_genital_organs', ('C60','C61','C62','C63'))

    # ## 攝護腺
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_prostate_gland', ('C61','185'))

    # ## 睪丸
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_testis', ('C62','186'))

    # ## 陰莖及其他男性生殖器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_penis_and_other_male_genital_organs', ('C60','C61','C62','C63','187'))

    # ## 泌尿器官
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_urinary_tract', ('C64','C65','C66','C67','C68'))

    # ## 膀胱
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_bladder', ('C67','188'))

    # ## 腎
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_kidney', ('C64','1890'))

    # ## 腎盂及其他泌尿系統
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_renal_pelvis', ('C65','C66','C68','1891','1892','1893','1894','1895','1896','1897','1898','1899'))

    # ## 眼及淚腺
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_eye_and_lacrimal_gland', ('C69','190'))

    # ## 中樞神經系統
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_brain_and_other_parts_of_central_nervous_system', ('C70','C71','C72','191','192'))

    # ## 腦
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_brain', ('C71','191'))

    # ## 腦膜、脊髓及神經系統未詳細說明部位
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system', ('C70','C72','192'))

    # ## 甲狀腺
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_thyroid_gland', ('C73','193'))

    # ## 其他內分泌腺
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_other_endocrine_glands', ('C74','C75','194'))

    # ## 不明原發部位
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_unknown_primary_sites', ('C80','199'))

    # ## 白血病
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_leukemia', ('M980','M981','M982','M983','M984','M985','M986','M987','M988','M989','M990','M991','M992','M993','M994','M995','M996','M997','M998','M999','204','205','206','207','208','C91','C92','C93','C94','C95'))

    # ## 惡性淋巴瘤
    df_TOTFA_sult = TOTFA_onehot('AO_Mal_malignant_lymphoma', ('M959','M960','M961','M962','M963','M964','M965','M966','M967','M968','M969','M970','M971','M972','M973','M974','M975','M976'))

    # ## 糖尿病Cohort
    df_TOTFA_sult = TOTFA_onehot('AO_Diabetes', ('E08','E09','E10','E11','E12','E13','250'))

    # ## 阿茲海默氏病Cohort
    df_TOTFA_sult = TOTFA_onehot('AO_Alzheimer', ('G30','3310'))

    # ## 慢性腎臟病Cohort
    df_TOTFA_sult = TOTFA_onehot('AO_CKD', ('A1811','A5275','C64','C65','C689','D300','D3A093','D410','D411','D422','D593','E082','E092','E1021','E1022','E1029','E1065','E112','E1165','E132','E260','E261','E268','E269','E7203','E723','E728','E740','E744','E748','E7521','E7522','E7524','E753','E77','I12','I13','I701','I722','I7581','I773','I7773','K767','M103','M1A10','M1A10','M1A1110','M1A1111','M1A1120','M1A1121','M1A1190','M1A1191','M1A1210','M1A1211','M1A1220','M1A1221','M1A1290','M1A1291','M1A1310','M1A1311','M1A1320','M1A1321','M1A1390','M1A1391','M1A1410','M1A1411','M1A1420','M1A1421','M1A1490','M1A1491','M1A1510','M1A1511','M1A1520','M1A1521','M1A1590','M1A1591','M1A1610','M1A1611','M1A1620','M1A1621','M1A1690','M1A1691','M1A1710','M1A1711','M1A1720','M1A1721','M1A1790','M1A1791','M1A18','M1A18','M1A19','M1A19','M3214','M3215','M3504','N00','N008','N009','N01','N02','N03','N04','N05','N06','N07','N08','N131','N132','N133','N135','N139','N14','N150','N158','N159','N16','N17','N18','N19','N200','N25','N261','N269','N271','N279','N289','N29','O104','O121','O2683','Q60','Q6102','Q611','Q612','Q613','Q614','Q615','Q618','Q620','Q621','Q622','Q6231','Q6232','Q6239','Q63','Q851','R944','T560','T560','T560','T560','Z4822','Z524','Z940','Z992','01600','01601','01602','01603','01604','01605','01606','0954','1890','1891','1899','2230','23691','25040','25041','25042','25043','2551','2708','2710','2714','2727','27410','27411','27419','28311','40300','40301','40310','40311','40390','40391','40400','40401','40402','40403','40410','40411','40412','40413','40490','40491','40492','40493','4401','4421','4473','5724','5800','5804','58081','58089','5809','5810','5811','5812','5813','58181','58189','5819','5820','5821','5822','5823','5824','58281','58289','5829','5830','5831','5832','5834','5836','5837','58381','58389','5839','5845','5846','5847','5848','5849','585','586','587','5880','5881','5888','5889','5891','5899','591','5933','5939','5996','64210','64211','64212','64213','64214','64620','64621','64622','64623','64624','7530','75312','75313','75314','75315','75316','75317','75319','75320','75321','75322','75329','7533','7595','7944','9849','V420','V451','V594'))
    
    #取唯一筆
    df_TOTFA_sult_final = df_TOTFA_sult[['id','AO_Myocardial_infarction','AO_Mal_larynx','AO_Congestive_heart_failure','AO_Peripheral_vascular_disease','AO_Cerebrovascular_disease','AO_Dementia','AO_Chronic_pulmonary_disease','AO_Rheumatic_disease','AO_Peptic_ulcer_disease','AO_Liver_disease_mild','AO_Diabetes_without_chronic_complications','AO_Renal_disease_mild_to_moderate','AO_Diabetes_with_chronic_complications','AO_Hemiplegia_or_paraplegia','AO_Any_malignancy','AO_Liver_disease_moderate_to_severe','AO_Renal_disease_severe','AO_HIV_infection_no_AIDS','AO_Metastatic_solid_tumor','AO_AIDS','AO_Mal_lip_oral_cavity_and_pharynx','AO_Mal_oral_cavity','AO_Mal_lip','AO_Mal_tongue','AO_Mal_gum','AO_Mal_floor_of_mouth','AO_Mal_other_and_unspecified_parts_of_mouth','AO_Mal_oropharynx','AO_Mal_hypopharynx','AO_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx','AO_Mal_major_salivary_glands','AO_Mal_nasopharynx','AO_Mal_digestive_organs_and_peritoneum','AO_Mal_esophagus','AO_Mal_stomach','AO_Mal_small_intestine','AO_Mal_colon_rectum_rectosigmoid_junction_and_anus','AO_Mal_colon','AO_Mal_rectum_rectosigmoid_junction_and_anus','AO_Mal_liver_and_intrahepatic_bile_ducts','AO_Mal_gallbladder_and_extrahepatic_bile_ducts','AO_Mal_pancreas','AO_Mal_retroperitoneum_and_peritoneum','AO_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum','AO_Mal_respiratory_system_and_intrathoracic_organs','AO_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear','AO_Mal_trachea_bronchus_and_lung','AO_Mal_pleura','AO_Mal_thymus_heart_and_mediastinum','AO_Mal_bones_joints_and_articular_cartilage','AO_Mal_connective_subcutaneous_and_other_soft_tissues','AO_Mal_skin','AO_Mal_breast','AO_Mal_female_genital_organs','AO_Mal_cervix_uteri','AO_Mal_corpus_uteri','AO_Mal_ovary_fallopian_tube_and_broad_ligament','AO_Mal_vagina_vulva_other_and_unspecified_female_genital_organs','AO_Mal_male_genital_organs','AO_Mal_prostate_gland','AO_Mal_testis','AO_Mal_penis_and_other_male_genital_organs','AO_Mal_urinary_tract','AO_Mal_bladder','AO_Mal_kidney','AO_Mal_renal_pelvis','AO_Mal_eye_and_lacrimal_gland','AO_Mal_brain_and_other_parts_of_central_nervous_system','AO_Mal_brain','AO_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system','AO_Mal_thyroid_gland','AO_Mal_other_endocrine_glands','AO_Mal_unknown_primary_sites','AO_Mal_leukemia','AO_Mal_malignant_lymphoma','AO_Diabetes','AO_Alzheimer','AO_CKD']]
    df_TOTFA_sult_final = df_TOTFA_sult_final.drop_duplicates(subset = 'id')
    TOTFA_cci_onehot = df_TOTFA_sult_final.copy()

    ############################################### TOTFA #############################################################

                                            ## CCI - count  ##
    def TOTFA_count(disease_Name, ICD):
        global r_e_combine2,TOTFA_cci_count,TOTFA_order_code,LABM1_order_code
        _19 = df_TOTFAE.loc[df_TOTFAE['d19'].str.startswith((ICD), na = False)]
        _20 = df_TOTFAE.loc[df_TOTFAE['d20'].str.startswith((ICD), na = False)]
        _21 = df_TOTFAE.loc[df_TOTFAE['d21'].str.startswith((ICD), na = False)]
        _22 = df_TOTFAE.loc[df_TOTFAE['d22'].str.startswith((ICD), na = False)]
        _23 = df_TOTFAE.loc[df_TOTFAE['d23'].str.startswith((ICD), na = False)]
        
        col_combine = pd.concat([_19,_20,_21,_22,_23])
        
        early_time = col_combine.drop_duplicates(subset = 'id')
        early_time = early_time[['id','t2','d9']]
        early_time.rename(columns={'t2': disease_Name+'_t2','d9':disease_Name + '_d9'}, inplace=True)
        
        col_combine = col_combine.drop_duplicates(subset = ['verify'])
        row_count = col_combine['id'].value_counts() 
        row_count = row_count.to_frame()
        row_count = row_count.reset_index()
        row_count.rename(columns={'id': disease_Name,'index':'id'}, inplace=True)
        r_e_combine2 = pd.merge(row_count, early_time, how='outer', on=['id'], indicator=False).fillna(value=0)
        return r_e_combine2

    df_TOTFA_sult_final2_0 = TOTFA_count('AC_Myocardial_infarction', ('410','412','I21','I22','I252'))
    df_TOTFA_sult_final2_1 = TOTFA_count('AC_Congestive_heart_failure', ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'))
    df_TOTFA_sult_final2_2 = TOTFA_count('AC_Peripheral_vascular_disease', ('0930','4373','440','441','4431','4432','4438','4439','4471','5571','5579','V434','I70','I71','I731','I738','I739','I771','I790','I791','I798','K551','K558','K559','Z958','Z959'))
    df_TOTFA_sult_final2_3 = TOTFA_count('AC_Cerebrovascular_disease', ('36234','430','431','432','433','434','435','436','437','438','G45','G46','H340','H341','H342','I60','I61','I62','I63','I64','I65','I66','I67','I68'))
    df_TOTFA_sult_final2_4 = TOTFA_count('AC_Dementia', ('2900','2901','2902','2903','2904','2940','2941','2942','2948','3310','3311','3312','3317','797','F01','F02','F03','F04','F05','F061','F068','G132','G138','G30','G310','G311','G312','G914','G94','R4181','R54'))
    df_TOTFA_sult_final2_5 = TOTFA_count('AC_Chronic_pulmonary_disease', ('490','491','492','493','494','495','496','500','501','502','503','504','505','5064','5081','5088','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703'))
    df_TOTFA_sult_final2_6 = TOTFA_count('AC_Rheumatic_disease', ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148','725','M05','M06','M315','M32','M33','M34','M351','M353','M360'))
    df_TOTFA_sult_final2_7 = TOTFA_count('AC_Peptic_ulcer_disease', ('531','532','533','534','K25','K26','K27','K28'))
    df_TOTFA_sult_final2_8 = TOTFA_count('AC_Liver_disease_mild', ('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427','B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944'))
    df_TOTFA_sult_final2_9 = TOTFA_count('AC_Diabetes_without_chronic_complications', ('2508','2509','2490','2491','2492','2493','2499','E080','E090','E100','E110','E130','E081','E091','E101','E111','E131','E086','E096','E106','E116','E136','E088','E098','E108','E118','E138','E089','E099','E109','E119','E139'))
    df_TOTFA_sult_final2_10 = TOTFA_count('AC_Renal_disease_mild_to_moderate', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))
    df_TOTFA_sult_final2_11 = TOTFA_count('AC_Diabetes_with_chronic_complications', ('40300','40310','40390','40400','40401','40410','40411','40490','40491','582','583','5851','5852','5853','5854','5859','V420','I129','I130','I1310','N03','N05','N181','N182','N183','N184','N189','Z940'))
    df_TOTFA_sult_final2_12 = TOTFA_count('AC_Hemiplegia_or_paraplegia', ('3341','342','343','344','G041','G114','G800','G801','G802','G81','G82','G83'))
    df_TOTFA_sult_final2_13 = TOTFA_count('AC_Any_malignancy', ('14',' 15',' 16',' 170',' 171',' 172',' 174',' 175',' 176',' 179',' 18',' 190',' 191',' 192',' 193',' 194',' 195',' 1991',' 200',' 201',' 202',' 203',' 204',' 205',' 206',' 207',' 208',' 2386','C0','C1','C2','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','58','C60','C61','C62','C63','63','C76','C801','C81','C82','C83','C84','C85','C88','C9'))
    df_TOTFA_sult_final2_14 = TOTFA_count('AC_Liver_disease_moderate_to_severe', ('4560','4561','4562x','5722','5723','5724','5728','I850','I864','K704','K711','K721','K729','K765','K766','K767'))
    df_TOTFA_sult_final2_15 = TOTFA_count('AC_Renal_disease_severe', ('40301','40311','40391','40402','40403','40412','40413','40492','40493','5855','5856','586','5880','V4511','V4512','V560','V561','V562','V5631','V5632','V568','I120','I1311','I132','N185','N186','N19','N250','Z49','Z992'))
    df_TOTFA_sult_final2_16 = TOTFA_count('AC_HIV_infection_no_AIDS', ('042','B20'))
    df_TOTFA_sult_final2_17 = TOTFA_count('AC_Metastatic_solid_tumor', ('196','197','198','1990','C77','C78','C79','C800','C802'))
    df_TOTFA_sult_final2_18 = TOTFA_count('AC_AIDS', ('112','180','114','1175','0074','0785','3483','054','115','0072','176','200','201','202','203','204','205','206','207','208','209','031','010','011','012','013','014','015','016','017','018','018','1363','V1261','0463','0031','130','7994','B37','C53','B38','B45','A072','B25','G934','B00','B39','A073','C46','C81','C82','C83','C84','C85','C86','C87','C88','C89','C90','C91','C92','C93','C94','C95','C96','C96','A31','A15','A16','A17','A18','A19','B59','Z8701','A812','A021','B58','R64'))
    df_TOTFA_sult_final2_19 = TOTFA_count('AC_Mal_lip_oral_cavity_and_pharynx', ('C00','C01','C02','C03','C04','C05','C06','C09','C10','C12','C13','C14','140','141','143','144','145','146','148','149'))
    df_TOTFA_sult_final2_20 = TOTFA_count('AC_Mal_oral_cavity', ('C00','C01','C02','C03','C04','C05','C06','140','141','143','144','145'))
    df_TOTFA_sult_final2_21 = TOTFA_count('AC_Mal_lip', ('C00'))
    df_TOTFA_sult_final2_22 = TOTFA_count('AC_Mal_tongue', ('C01','C02'))
    df_TOTFA_sult_final2_23 = TOTFA_count('AC_Mal_gum', ('C03'))
    df_TOTFA_sult_final2_24 = TOTFA_count('AC_Mal_floor_of_mouth', ('C04'))
    df_TOTFA_sult_final2_25 = TOTFA_count('AC_Mal_other_and_unspecified_parts_of_mouth', ('C05','C06'))
    df_TOTFA_sult_final2_26 = TOTFA_count('AC_Mal_oropharynx', ('C09','C10'))
    df_TOTFA_sult_final2_27 = TOTFA_count('AC_Mal_hypopharynx', ('C12','C13'))
    df_TOTFA_sult_final2_28 = TOTFA_count('AC_Mal_pharynx_and_ill_defined_sites_in_lip_oral_cavity_and_pharynx', ('C14'))
    df_TOTFA_sult_final2_29 = TOTFA_count('AC_Mal_major_salivary_glands', ('C07','C08','142'))
    df_TOTFA_sult_final2_30 = TOTFA_count('AC_Mal_nasopharynx', ('C11','147'))
    df_TOTFA_sult_final2_31 = TOTFA_count('AC_Mal_digestive_organs_and_peritoneum', ('C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C48'))
    df_TOTFA_sult_final2_32 = TOTFA_count('AC_Mal_esophagus', ('C15','150'))
    df_TOTFA_sult_final2_33 = TOTFA_count('AC_Mal_stomach', ('C16','151'))
    df_TOTFA_sult_final2_34 = TOTFA_count('AC_Mal_small_intestine', ('C17','152'))
    df_TOTFA_sult_final2_35 = TOTFA_count('AC_Mal_colon_rectum_rectosigmoid_junction_and_anus', ('C18','C19','C20','C21','153','154'))
    df_TOTFA_sult_final2_36 = TOTFA_count('AC_Mal_colon', ('C18','153'))
    df_TOTFA_sult_final2_37 = TOTFA_count('AC_Mal_rectum_rectosigmoid_junction_and_anus', ('C19','C20','C21','154'))
    df_TOTFA_sult_final2_38 = TOTFA_count('AC_Mal_liver_and_intrahepatic_bile_ducts', ('C22','155'))
    df_TOTFA_sult_final2_39 = TOTFA_count('AC_Mal_gallbladder_and_extrahepatic_bile_ducts', ('C23','C24','156'))
    df_TOTFA_sult_final2_40 = TOTFA_count('AC_Mal_pancreas', ('C25','157'))
    df_TOTFA_sult_final2_41 = TOTFA_count('AC_Mal_retroperitoneum_and_peritoneum', ('C48','158'))
    df_TOTFA_sult_final2_42 = TOTFA_count('AC_Mal_other_and_ill_defined_sites_within_digestive_organs_and_peritoneum', ('C26','159'))
    df_TOTFA_sult_final2_43 = TOTFA_count('AC_Mal_respiratory_system_and_intrathoracic_organs', ('C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','160','161','162','163','164'))
    df_TOTFA_sult_final2_44 = TOTFA_count('AC_Mal_nasal_cavities_accessory_sinuses_middle_ear_and_inner_ear', ('C30','C31','160'))
    df_TOTFA_sult_final2_45 = TOTFA_count('AC_Mal_larynx', ('C32','161'))
    df_TOTFA_sult_final2_46 = TOTFA_count('AC_Mal_trachea_bronchus_and_lung', ('C33','C34','162'))
    df_TOTFA_sult_final2_47 = TOTFA_count('AC_Mal_pleura', ('C384','163'))
    df_TOTFA_sult_final2_48 = TOTFA_count('AC_Mal_thymus_heart_and_mediastinum', ('C37','C380','C381','C382','C383','C388','164'))
    df_TOTFA_sult_final2_49 = TOTFA_count('AC_Mal_bones_joints_and_articular_cartilage', ('C40','C41','170'))
    df_TOTFA_sult_final2_50 = TOTFA_count('AC_Mal_connective_subcutaneous_and_other_soft_tissues', ('C47','C49','171'))
    df_TOTFA_sult_final2_51 = TOTFA_count('AC_Mal_skin', ('C43','C44','172','173'))
    df_TOTFA_sult_final2_52 = TOTFA_count('AC_Mal_breast', ('C50','174','175','C50'))
    df_TOTFA_sult_final2_53 = TOTFA_count('AC_Mal_female_genital_organs', ('C51','C52','C53','C54','C55','C56','C57','C58'))
    df_TOTFA_sult_final2_54 = TOTFA_count('AC_Mal_cervix_uteri', ('C53','C55','179','180'))
    df_TOTFA_sult_final2_55 = TOTFA_count('AC_Mal_corpus_uteri', ('C54','182'))
    df_TOTFA_sult_final2_56 = TOTFA_count('AC_Mal_ovary_fallopian_tube_and_broad_ligament', ('C56','C570','C571','C572','C573','C574','183'))
    df_TOTFA_sult_final2_57 = TOTFA_count('AC_Mal_vagina_vulva_other_and_unspecified_female_genital_organs', ('C51','C52','C577','C578','C579','184'))
    df_TOTFA_sult_final2_58 = TOTFA_count('AC_Mal_male_genital_organs', ('C60','C61','C62','C63'))
    df_TOTFA_sult_final2_59 = TOTFA_count('AC_Mal_prostate_gland', ('C61','185'))
    df_TOTFA_sult_final2_60 = TOTFA_count('AC_Mal_testis', ('C62','186'))
    df_TOTFA_sult_final2_61 = TOTFA_count('AC_Mal_penis_and_other_male_genital_organs', ('C60','C61','C62','C63','187'))
    df_TOTFA_sult_final2_62 = TOTFA_count('AC_Mal_urinary_tract', ('C64','C65','C66','C67','C68'))
    df_TOTFA_sult_final2_63 = TOTFA_count('AC_Mal_bladder', ('C67','188'))
    df_TOTFA_sult_final2_64 = TOTFA_count('AC_Mal_kidney', ('C64','1890'))
    df_TOTFA_sult_final2_65 = TOTFA_count('AC_Mal_renal_pelvis', ('C65','C66','C68','1891','1892','1893','1894','1895','1896','1897','1898','1899'))
    df_TOTFA_sult_final2_66 = TOTFA_count('AC_Mal_eye_and_lacrimal_gland', ('C69','190'))
    df_TOTFA_sult_final2_67 = TOTFA_count('AC_Mal_brain_and_other_parts_of_central_nervous_system', ('C70','C71','C72','191','192'))
    df_TOTFA_sult_final2_68 = TOTFA_count('AC_Mal_brain', ('C71','191'))
    df_TOTFA_sult_final2_69 = TOTFA_count('AC_Mal_meninges_spinal_cord_and_other_parts_of_central_nervous_system', ('C70','C72','192'))
    df_TOTFA_sult_final2_70 = TOTFA_count('AC_Mal_thyroid_gland', ('C73','193'))
    df_TOTFA_sult_final2_71 = TOTFA_count('AC_Mal_other_endocrine_glands', ('C74','C75','194'))
    df_TOTFA_sult_final2_72 = TOTFA_count('AC_Mal_unknown_primary_sites', ('C80','199'))
    df_TOTFA_sult_final2_73 = TOTFA_count('AC_Mal_leukemia', ('M980','M981','M982','M983','M984','M985','M986','M987','M988','M989','M990','M991','M992','M993','M994','M995','M996','M997','M998','M999','204','205','206','207','208','C91','C92','C93','C94','C95'))
    df_TOTFA_sult_final2_74 = TOTFA_count('AC_Mal_malignant_lymphoma', ('M959','M960','M961','M962','M963','M964','M965','M966','M967','M968','M969','M970','M971','M972','M973','M974','M975','M976'))
    df_TOTFA_sult_final2_75 = TOTFA_count('AC_Diabetes', ('E08','E09','E10','E11','E12','E13','250'))
    df_TOTFA_sult_final2_76 = TOTFA_count('AC_Alzheimer', ('G30','3310'))
    df_TOTFA_sult_final2_77 = TOTFA_count('AC_CKD', ('A1811','A5275','C64','C65','C689','D300','D3A093','D410','D411','D422','D593','E082','E092','E1021','E1022','E1029','E1065','E112','E1165','E132','E260','E261','E268','E269','E7203','E723','E728','E740','E744','E748','E7521','E7522','E7524','E753','E77','I12','I13','I701','I722','I7581','I773','I7773','K767','M103','M1A10','M1A10','M1A1110','M1A1111','M1A1120','M1A1121','M1A1190','M1A1191','M1A1210','M1A1211','M1A1220','M1A1221','M1A1290','M1A1291','M1A1310','M1A1311','M1A1320','M1A1321','M1A1390','M1A1391','M1A1410','M1A1411','M1A1420','M1A1421','M1A1490','M1A1491','M1A1510','M1A1511','M1A1520','M1A1521','M1A1590','M1A1591','M1A1610','M1A1611','M1A1620','M1A1621','M1A1690','M1A1691','M1A1710','M1A1711','M1A1720','M1A1721','M1A1790','M1A1791','M1A18','M1A18','M1A19','M1A19','M3214','M3215','M3504','N00','N008','N009','N01','N02','N03','N04','N05','N06','N07','N08','N131','N132','N133','N135','N139','N14','N150','N158','N159','N16','N17','N18','N19','N200','N25','N261','N269','N271','N279','N289','N29','O104','O121','O2683','Q60','Q6102','Q611','Q612','Q613','Q614','Q615','Q618','Q620','Q621','Q622','Q6231','Q6232','Q6239','Q63','Q851','R944','T560','T560','T560','T560','Z4822','Z524','Z940','Z992','01600','01601','01602','01603','01604','01605','01606','0954','1890','1891','1899','2230','23691','25040','25041','25042','25043','2551','2708','2710','2714','2727','27410','27411','27419','28311','40300','40301','40310','40311','40390','40391','40400','40401','40402','40403','40410','40411','40412','40413','40490','40491','40492','40493','4401','4421','4473','5724','5800','5804','58081','58089','5809','5810','5811','5812','5813','58181','58189','5819','5820','5821','5822','5823','5824','58281','58289','5829','5830','5831','5832','5834','5836','5837','58381','58389','5839','5845','5846','5847','5848','5849','585','586','587','5880','5881','5888','5889','5891','5899','591','5933','5939','5996','64210','64211','64212','64213','64214','64620','64621','64622','64623','64624','7530','75312','75313','75314','75315','75316','75317','75319','75320','75321','75322','75329','7533','7595','7944','9849','V420','V451','V594'))

    list_cci = [df_TOTFA_sult_final2_0 ,df_TOTFA_sult_final2_1 ,df_TOTFA_sult_final2_2 ,df_TOTFA_sult_final2_3 ,df_TOTFA_sult_final2_4 ,df_TOTFA_sult_final2_5 ,df_TOTFA_sult_final2_6 ,df_TOTFA_sult_final2_7 ,df_TOTFA_sult_final2_8 ,df_TOTFA_sult_final2_9, df_TOTFA_sult_final2_10, df_TOTFA_sult_final2_11, df_TOTFA_sult_final2_12, df_TOTFA_sult_final2_13, df_TOTFA_sult_final2_14, df_TOTFA_sult_final2_15, df_TOTFA_sult_final2_16, df_TOTFA_sult_final2_17, df_TOTFA_sult_final2_18, df_TOTFA_sult_final2_19, df_TOTFA_sult_final2_20, df_TOTFA_sult_final2_21, df_TOTFA_sult_final2_22, df_TOTFA_sult_final2_23, df_TOTFA_sult_final2_24, df_TOTFA_sult_final2_25, df_TOTFA_sult_final2_26, df_TOTFA_sult_final2_27, df_TOTFA_sult_final2_28, df_TOTFA_sult_final2_29, df_TOTFA_sult_final2_30, df_TOTFA_sult_final2_31, df_TOTFA_sult_final2_32, df_TOTFA_sult_final2_33, df_TOTFA_sult_final2_34, df_TOTFA_sult_final2_35, df_TOTFA_sult_final2_36, df_TOTFA_sult_final2_37, df_TOTFA_sult_final2_38, df_TOTFA_sult_final2_39, df_TOTFA_sult_final2_40, df_TOTFA_sult_final2_41, df_TOTFA_sult_final2_42, df_TOTFA_sult_final2_43, df_TOTFA_sult_final2_44, df_TOTFA_sult_final2_45, df_TOTFA_sult_final2_46, df_TOTFA_sult_final2_47, df_TOTFA_sult_final2_48, df_TOTFA_sult_final2_49, df_TOTFA_sult_final2_50, df_TOTFA_sult_final2_51, df_TOTFA_sult_final2_52, df_TOTFA_sult_final2_53, df_TOTFA_sult_final2_54, df_TOTFA_sult_final2_55, df_TOTFA_sult_final2_56, df_TOTFA_sult_final2_57, df_TOTFA_sult_final2_58, df_TOTFA_sult_final2_59, df_TOTFA_sult_final2_60, df_TOTFA_sult_final2_61, df_TOTFA_sult_final2_62, df_TOTFA_sult_final2_63, df_TOTFA_sult_final2_64, df_TOTFA_sult_final2_65, df_TOTFA_sult_final2_66, df_TOTFA_sult_final2_67, df_TOTFA_sult_final2_68, df_TOTFA_sult_final2_69, df_TOTFA_sult_final2_70, df_TOTFA_sult_final2_71, df_TOTFA_sult_final2_72, df_TOTFA_sult_final2_73, df_TOTFA_sult_final2_74, df_TOTFA_sult_final2_75, df_TOTFA_sult_final2_76, df_TOTFA_sult_final2_77]

    i=0
    for loop in range(len(list_cci)-1):
        if i<1:
            A_loop_table = pd.merge(list_cci[1], list_cci[0], how='outer', on=['id'], indicator=False).fillna(value=0)
            i+=2
            # print('if',i)
        else:
            A_loop_table = pd.merge(list_cci[i], A_loop_table, how='outer', on=['id'], indicator=False).fillna(value=0)
            i+=1
            # print('else',i)
            
    # A_loop_table = A_loop_table[['id','AC_Any_malignancy','AC_Any_malignancy_t2','AC_Any_malignancy_d9','AC_Metastatic_solid_tumor','AC_Metastatic_solid_tumor_t2','AC_Metastatic_solid_tumor_d9','AC_AIDS','AC_AIDS_t2','AC_AIDS_d9','AC_HIV_infection_no_AIDS','AC_HIV_infection_no_AIDS_t2','AC_HIV_infection_no_AIDS_d9','AC_Renal_disease_severe','AC_Renal_disease_severe_t2','AC_Renal_disease_severe_d9','AC_Liver_disease_moderate_to_severe','AC_Liver_disease_moderate_to_severe_t2','AC_Liver_disease_moderate_to_severe_d9','AC_Hemiplegia_or_paraplegia','AC_Hemiplegia_or_paraplegia_t2','AC_Hemiplegia_or_paraplegia_d9','AC_Renal_disease_mild_to_moderate','AC_Renal_disease_mild_to_moderate_t2','AC_Renal_disease_mild_to_moderate_d9','AC_Diabetes_without_chronic_complications','AC_Diabetes_without_chronic_complications_t2','AC_Diabetes_without_chronic_complications_d9','AC_Diabetes_with_chronic_complications','AC_Diabetes_with_chronic_complications_t2','AC_Diabetes_with_chronic_complications_d9','AC_Liver_disease_mild', 'AC_Liver_disease_mild_t2','AC_Liver_disease_mild_d9','AC_Peptic_ulcer_disease','AC_Peptic_ulcer_disease_t2','AC_Peptic_ulcer_disease_d9','AC_Rheumatic_disease','AC_Rheumatic_disease_t2','AC_Rheumatic_disease_d9','AC_Chronic_pulmonary_disease','AC_Chronic_pulmonary_disease_t2','AC_Chronic_pulmonary_disease_d9','AC_Dementia','AC_Dementia_t2','AC_Dementia_d9','AC_Cerebrovascular_disease','AC_Cerebrovascular_disease_t2','AC_Cerebrovascular_disease_d9','AC_Peripheral_vascular_disease','AC_Peripheral_vascular_disease_t2','AC_Peripheral_vascular_disease_d9','AC_Congestive_heart_failure','AC_Congestive_heart_failure_t2','AC_Congestive_heart_failure_d9','AC_Myocardial_infarction','AC_Myocardial_infarction_t2','AC_Myocardial_infarction_d9',
    #                              'AC_Mal_lip_oral_pharynx', 'AC_Mal_lip_oral_pharynx_t2', 'AC_Mal_lip_oral_pharynx_d9', 'AC_Mal_digestive_peritoneum', 'AC_Mal_digestive_peritoneum_t2', 'AC_Mal_digestive_peritoneum_d9', 'AC_Mal_respiratory_intrathoracic', 'AC_Mal_respiratory_intrathoracic_t2', 'AC_Mal_respiratory_intrathoracic_d9', 'AC_Mal_bone_articular', 'AC_Mal_bone_articular_t2', 'AC_Mal_bone_articular_d9', 'AC_Mal_skin', 'AC_Mal_skin_t2', 'AC_Mal_skin_d9', 'AC_Mal_mesothelial_soft_tissues', 'AC_Mal_mesothelial_soft_tissues_t2', 'AC_Mal_mesothelial_soft_tissues_d9', 'AC_Mal_breast', 'AC_Mal_breast_t2', 'AC_Mal_breast_d9', 'AC_Mal_female_reproductive', 'AC_Mal_female_reproductive_t2', 'AC_Mal_female_reproductive_d9', 'AC_Mal_male_reproductive', 'AC_Mal_male_reproductive_t2', 'AC_Mal_male_reproductive_d9', 'AC_Mal_urinary', 'AC_Mal_urinary_t2', 'AC_Mal_urinary_d9', 'AC_Mal_eye_brain_central_nervous', 'AC_Mal_eye_brain_central_nervous_t2', 'AC_Mal_eye_brain_central_nervous_d9', 'AC_Mal_thyroid_endocrine_glands', 'AC_Mal_thyroid_endocrine_glands_t2', 'AC_Mal_thyroid_endocrine_glands_d9', 'AC_Mal_Unspecified_sec', 'AC_Mal_Unspecified_sec_t2', 'AC_Mal_Unspecified_sec_d9', 'AC_Mal_lymphoid_hema_tissues', 'AC_Mal_lymphoid_hema_tissues_t2', 'AC_Mal_lymphoid_hema_tissues_d9', 'AC_Mal_Independent_mult', 'AC_Mal_Independent_mult_t2', 'AC_Mal_Independent_mult_d9', 'AC_Mal_Carcinoma_in_situ', 'AC_Mal_Carcinoma_in_situ_t2', 'AC_Mal_Carcinoma_in_situ_d9', 'AC_Mal_Benign_tumor', 'AC_Mal_Benign_tumor_t2', 'AC_Mal_Benign_tumor_d9', 'AC_Mal_unknown', 'AC_Mal_unknown_t2', 'AC_Mal_unknown_d9']]
    TOTFA_cci_count = A_loop_table.copy()

                                                ## A_order_code - count ##
    A_order_code = df_TOTFAO1[['id','p4']]
    total_counts = A_order_code.groupby(['id'])
    total_counts = total_counts.size().reset_index(name='A_p4_total_counts')

    A_order_code = df_TOTFAO1.drop_duplicates(subset = ['id','p4'])
    unique_counts = A_order_code.groupby(['id'])
    unique_counts = unique_counts.size().reset_index(name='A_p4_unique_counts')

    total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    TOTFA_order_code = total_unique_table.copy()

    ############################################### LABM1 #############################################################                                           
                                      ## LABM1_order_code - count ##
    global LABM1_order_code,LABM2_order_code,LABD1_order_code,LABD2_order_code,combine_output
    LABM1_order_code = df_LABM1[['id','h18']]
    total_counts = LABM1_order_code.groupby(["id"])
    total_counts = total_counts.size().reset_index(name='LAB1_h18_total_counts')

    LABM1_order_code = LABM1_order_code.drop_duplicates(subset = ['id','h18'])
    unique_counts = LABM1_order_code.groupby(["id"])
    unique_counts = unique_counts.size().reset_index(name='LAB1_h18_unique_counts')

    total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    LABM1_order_code = total_unique_table.copy()
    ############################################### LABM2 #############################################################                                           
                                      ## LABM2_order_code - count ##
    LABM2_order_code = df_LABM2[['id','h18']]
    total_counts = LABM2_order_code.groupby(["id"])
    total_counts = total_counts.size().reset_index(name='LAB2_h18_total_counts')

    LABM2_order_code = LABM2_order_code.drop_duplicates(subset = ['id','h18'])
    unique_counts = LABM1_order_code.groupby(["id"])
    unique_counts = unique_counts.size().reset_index(name='LAB2_h18_unique_counts')

    total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    LABM2_order_code = total_unique_table.copy()
    ############################################### LABD1 #############################################################                                           
                                      ## LABD1_order_code - count ##
    # LABD1_order_code = df_LABD1[['id','h15']]
    # total_counts = LABD1_order_code.groupby(["id"])
    # total_counts = total_counts.size().reset_index(name='LABD1_h15_total_counts')

    # LABD1_order_code = LABD1_order_code.drop_duplicates(subset = ['id','h15'])
    # unique_counts = LABD1_order_code.groupby(["id"])
    # unique_counts = unique_counts.size().reset_index(name='LABD1_h15_unique_counts')

    # total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    # LABD1_order_code = total_unique_table.copy()
    ############################################### LABD2 #############################################################                                           
                                      ## LABD2_order_code - count ##
    # LABD2_order_code = df_LABD2[['id','h15']]
    # total_counts = LABD2_order_code.groupby(["id"])
    # total_counts = total_counts.size().reset_index(name='LABD2_h15_total_counts')

    # LABD2_order_code = LABD2_order_code.drop_duplicates(subset = ['id','h15'])
    # unique_counts = LABD2_order_code.groupby(["id"])
    # unique_counts = unique_counts.size().reset_index(name='LABD2_h15_unique_counts')

    # total_unique_table = pd.merge(total_counts, unique_counts, how='outer', on=['id'], indicator=False).fillna(value=0)
    # LABD2_order_code = total_unique_table.copy()
    ############################################## table_count ########################################################

    # global CASE_count,CRLF_count,CRSF_count,DEATH_count,LABD1_count,LABD2_count,LABM1_count,LABM2_count,TOTFA_count,TOTFB_count,combine_count
    CASE_count = df_CASE.groupby(["id"])
    CASE_count = CASE_count.size().reset_index(name='CASE_total_counts')
        
    CRLF_count = df_CRLF.groupby(["id"])
    CRLF_count = CRLF_count.size().reset_index(name='CRLF_total_counts')
        
    # CRSF_count = df_CRSF.groupby(["id"])
    # CRSF_count = CRSF_count.size().reset_index(name='CRSF_total_counts')
    
    DEATH_count = df_DEATH.groupby(["id"])
    DEATH_count = DEATH_count.size().reset_index(name='DEATH_total_counts')        
    
    # LABD1_count = df_LABD1.groupby(["id"])
    # LABD1_count = LABD1_count.size().reset_index(name='LABD1_total_counts') 
    
    # LABD2_count = df_LABD2.groupby(["id"])
    # LABD2_count = LABD2_count.size().reset_index(name='LABD2_total_counts')
    
    LABM1_count = df_LABM1.groupby(["id"])
    LABM1_count = LABM1_count.size().reset_index(name='LAB1_total_counts')
    
    LABM2_count = df_LABM2.groupby(["id"])
    LABM2_count = LABM2_count.size().reset_index(name='LAB2_total_counts')
    
    TOTFAE_count = df_TOTFAE.groupby(["id"])
    TOTFAE_count = TOTFAE_count.size().reset_index(name='TOTFAE_total_counts')
    
    
    TOTFBE_count = df_TOTFBE.groupby(["id"])
    TOTFBE_count = TOTFBE_count.size().reset_index(name='TOTFBE_total_counts')
    
    TOTFAO1_count = df_TOTFAO1.groupby(["id"])
    TOTFAO1_count = TOTFAO1_count.size().reset_index(name='TOTFAO1_total_counts')

    TOTFAO2_count = df_TOTFAO2.groupby(["id"])
    TOTFAO2_count = TOTFAO2_count.size().reset_index(name='TOTFAO2_total_counts')

    TOTFBO1_count = df_TOTFBO1.groupby(["id"])
    TOTFBO1_count = TOTFBO1_count.size().reset_index(name='TOTFBO1_total_counts')
    
    TOTFBO2_count = df_TOTFBO2.groupby(["id"])
    TOTFBO2_count = TOTFBO2_count.size().reset_index(name='TOTFBO2_total_counts')

    combine_count = pd.merge(CASE_count, CRLF_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    # combine_count = pd.merge(CRSF_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(DEATH_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    # combine_count = pd.merge(LABD1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    # combine_count = pd.merge(LABD2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(LABM1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(LABM2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)        
    combine_count = pd.merge(TOTFAE_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(TOTFBE_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(TOTFAO1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(TOTFAO2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(TOTFBO1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_count = pd.merge(TOTFBO2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)

    ########################################################################
    #################################### 合併 ##############################
    #######################################################################
    
    combine_output = pd.merge(TOTFA_cci_count, TOTFA_cci_onehot, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(TOTFA_order_code, combine_output, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(TOTFB_cci_onehot, combine_output, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(TOTFB_cci_count, combine_output, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(TOTFB_order_code, combine_output, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(combine_output, LABM1_order_code, how='outer', on=['id'], indicator=False).fillna(value=0)
    combine_output = pd.merge(combine_output, LABM2_order_code, how='outer', on=['id'], indicator=False).fillna(value=0)
    # combine_output = pd.merge(combine_output, LABD1_order_code, how='outer', on=['id'], indicator=False).fillna(value=0)
    # combine_output = pd.merge(combine_output, LABD2_order_code, how='outer', on=['id'], indicator=False).fillna(value=0)

    def Form2_new_table():
        global Form2_new_table
        try:
            df_CRLF['age'] = df_CRLF['age'].astype(int)
        except:
            pass
        df_CRLF['age_group'] = df_CRLF['age'].apply(age_group)
        Form2_new_table = df_CRLF[['id','sex','resid','dbirth','age','age_group','site','didiag','grade_c','grade_p','cstage','pstage','vstatus','ssf1','ssf2','ssf3','ssf4','ssf5','ssf6','ssf7','ssf8','ssf9','ssf10']]
        Form2_new_table = Form2_new_table.drop_duplicates(subset = ["id","site","didiag"])
        Form2_new_table['dbirth'] = Form2_new_table['dbirth'].apply(fourdate)
        Form2_new_table['didiag'] = Form2_new_table['didiag'].apply(fourdate)
        return Form2_new_table

    #-合併CRLF挑的欄位 & 各表的計筆數
    global up,down,Form2_new_table_sult
    up = combine_output
    down = Form2_new_table()
    Form2_new_table_sult = pd.merge(down, up, how='outer', on=['id'], indicator=False).fillna(np.nan).replace([np.nan],[None])
    Form2_new_table_sult = pd.merge(Form2_new_table_sult, combine_count, how='outer', on=['id'], indicator=False)
    Form2_new_table_sult = insert_age_gender(Form2_new_table_sult)
    # des_Form2_new_table_sult = describe(Form2_new_table_sult).T
    # des_Form2_new_table_sult  = missing(des_Form2_new_table_sult, Form2_new_table_sult)
    # des_Form2_new_table_sult_v  = split_v_c(des_Form2_new_table_sult)[0]
    # des_Form2_new_table_sult_c  = split_v_c(des_Form2_new_table_sult)[1]
    end = time.perf_counter()
    print(end - start)
    # return(Form2_new_table_sult,des_Form2_new_table_sult_v,des_Form2_new_table_sult_c)
    # Form2_new_table_sult.to_json("./test.json", orient = 'records',date_format = 'iso', double_precision=3)
    
    conn.close()
    return(Form2_new_table_sult)

def B_plus_form2_tmp(ID):
    pass

def B_plus_form1(stats,logic):
    # today_date = "'"+time.strftime("%Y-%m-%d 00:00:00")+"'"
    def max_date(table):
        try:
            table_name = str(table)
            df = pd.read_sql("SELECT Max(ModifyTime) FROM " + "[" + table_name + "]" , conn)
            df['Max(ModifyTime)']= pd.to_datetime(df['Max(ModifyTime)']) - pd.Timedelta(days=1)
            df = df.astype(str)
            today_date = "'"+df.iat[0,0]+"'"
            return(today_date)
        except:
            return("'"+time.strftime("%Y-%m-%d 00:00:00")+"'")
    try:
        sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
        conn = sqlite3.connect(sqldb)
        cursor = conn.cursor()

        today_date_CASE = max_date('CASE')
        today_date_CRLF = max_date('CRLF')
        # today_date_CRSF = max_date('CRSF')
        today_date_DEATH = max_date('DEATH')
        # today_date_LABD1 = max_date('LABD1')
        # today_date_LABD2 = max_date('LABD2')
        today_date_LABM1 =  max_date('LABM1')
        today_date_LABM2 =  max_date('LABM2')
        today_date_TOTFAE = max_date('TOTFAE')
        today_date_TOTFAO1 = max_date('TOTFAO1')
        today_date_TOTFAO2 = max_date('TOTFAO2')
        today_date_TOTFBE = max_date('TOTFBE')
        today_date_TOTFBO1 = max_date('TOTFBO1')
        today_date_TOTFBO2 = max_date('TOTFBO2')        

        if len(logic)>3:

            logic_query = logic_json(logic)

            if len(logic_query)<3:
                return(ex_df,ex_df,ex_df)

        global df_CASE, df_CRLF, df_CRSF, df_DEATH, df_LABD1, df_LABD2, df_LABM1, df_LABM2, df_TOTFAE, df_TOTFAO1, df_TOTFAO2, df_TOTFBE, df_TOTFBO1, df_TOTFBO2
        
        if stats =='update':
            df_CASE = pd.read_sql("SELECT * FROM " + "[" + 'CASE' + "]" + "where ModifyTime>=" + today_date_CASE, conn)
            df_CRLF = pd.read_sql("SELECT * FROM " + "[" + 'CRLF' + "]" + "where ModifyTime>=" + today_date_CRLF, conn) #ID主檔 不能篩時間掉 否則會沒東西
            # df_CRSF = pd.read_sql("SELECT * FROM " + "[" + 'CRSF' + "]" + "where ModifyTime>=" + today_date_CRSF, conn)
            df_DEATH = pd.read_sql("SELECT * FROM " + "[" + 'DEATH' + "]" + "where ModifyTime>=" + today_date_DEATH, conn)
            # df_LABD1 = pd.read_sql("SELECT * FROM " + "[" + 'LABD1' + "]" + "where ModifyTime>=" + today_date_LABD1, conn)
            # df_LABD2 = pd.read_sql("SELECT * FROM " + "[" + 'LABD2' + "]" + "where ModifyTime>=" + today_date_LABD2, conn)
            df_LABM1 = pd.read_sql("SELECT * FROM " + "[" + 'LABM1' + "]" + "where ModifyTime>=" + today_date_LABM1, conn)
            df_LABM2 = pd.read_sql("SELECT * FROM " + "[" + 'LABM2' + "]" + "where ModifyTime>=" + today_date_LABM2, conn)
            df_TOTFAE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAE' + "]" + "where ModifyTime>=" + today_date_TOTFAE, conn)
            df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO1' + "]" + "where ModifyTime>=" + today_date_TOTFAO1, conn)
            df_TOTFAO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO2' + "]" + "where ModifyTime>=" + today_date_TOTFAO2, conn)
            df_TOTFBE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBE' + "]" + "where ModifyTime>=" + today_date_TOTFBE, conn)
            df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO1' + "]" + "where ModifyTime>=" + today_date_TOTFBO1, conn)
            df_TOTFBO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO2' + "]" + "where ModifyTime>=" + today_date_TOTFBO2, conn)
        else:
            df_CASE = pd.read_sql("SELECT * FROM " + "[" + 'CASE' + "]", conn)
            df_CRLF = pd.read_sql("SELECT * FROM " + "[" + 'CRLF' + "]", conn)
            # df_CRSF = pd.read_sql("SELECT * FROM " + "[" + 'CRSF' + "]", conn)
            df_DEATH = pd.read_sql("SELECT * FROM " + "[" + 'DEATH' + "]", conn)
            # df_LABD1 = pd.read_sql("SELECT * FROM " + "[" + 'LABD1' + "]", conn)
            # df_LABD2 = pd.read_sql("SELECT * FROM " + "[" + 'LABD2' + "]", conn)
            df_LABM1 = pd.read_sql("SELECT * FROM " + "[" + 'LABM1' + "]", conn)
            df_LABM2 = pd.read_sql("SELECT * FROM " + "[" + 'LABM2' + "]", conn)
            df_TOTFAE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAE' + "]", conn)
            df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO1' + "]", conn)
            df_TOTFAO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFAO2' + "]", conn)
            df_TOTFBE = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBE' + "]", conn)
            df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO1' + "]", conn)
            df_TOTFBO2 = pd.read_sql("SELECT * FROM " + "[" + 'TOTFBO2' + "]", conn)

        # df_CASE['ModifyTime'] = df_CASE['ModifyTime'].astype('datetime64[ns]')
        # df_CRLF['ModifyTime'] = df_CRLF['ModifyTime'].astype('datetime64[ns]')
        # df_CRSF['ModifyTime'] = df_CRSF['ModifyTime'].astype('datetime64[ns]')
        # df_DEATH['ModifyTime'] = df_DEATH['ModifyTime'].astype('datetime64[ns]')
        # df_LABD1['ModifyTime'] = df_LABD1['ModifyTime'].astype('datetime64[ns]')
        # df_LABD2['ModifyTime'] = df_LABD2['ModifyTime'].astype('datetime64[ns]')
        # df_LABM1['ModifyTime'] = df_LABM1['ModifyTime'].astype('datetime64[ns]')
        # df_LABM2['ModifyTime'] = df_LABM2['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFAE['ModifyTime'] = df_TOTFAE['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFAO1['ModifyTime'] = df_TOTFAO1['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFAO2['ModifyTime'] = df_TOTFAO2['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFBE['ModifyTime'] = df_TOTFBE['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFBO1['ModifyTime'] = df_TOTFBO1['ModifyTime'].astype('datetime64[ns]')
        # df_TOTFBO2['ModifyTime'] = df_TOTFBO2['ModifyTime'].astype('datetime64[ns]')
        # df_LABD1.rename(columns={'h9': 'id'}, inplace=True)
        # df_LABD2.rename(columns={'h9': 'id'}, inplace=True)
        df_LABM1.rename(columns={'h9': 'id'}, inplace=True)
        df_LABM2.rename(columns={'h9': 'id'}, inplace=True)
        df_TOTFAE.rename(columns={'d3': 'id'}, inplace=True)
        df_TOTFAO1.rename(columns={'d3': 'id'}, inplace=True)
        df_TOTFAO2.rename(columns={'d3': 'id'}, inplace=True)
        df_TOTFBE.rename(columns={'d3': 'id'}, inplace=True)
        df_TOTFBO1.rename(columns={'d3': 'id'}, inplace=True)
        df_TOTFBO2.rename(columns={'d3': 'id'}, inplace=True)

        def query_loop(dataframe):

            logic_query = logic_json(logic)
            match_df = pd.DataFrame()
            query_batch = logic_query.split('|')

            i=0
            for b in range(len(query_batch)):
                df = dataframe.query(query_batch[i])
                match_df = pd.concat([match_df, df], axis=0)
                i=i+1

            return(match_df)

        if len(logic) >= 5:
            logic_query = logic_json(logic)
            df_CASE = query_loop(df_CASE)
            df_CRLF = query_loop(df_CRLF) 
            # df_CRSF = query_loop(df_CRSF)
            df_DEATH = query_loop(df_DEATH)
            # df_LABD1 = query_loop(df_LABD1)
            # df_LABD2 = query_loop(df_LABD2)
            df_LABM1 = query_loop(df_LABM1)
            df_LABM2 = query_loop(df_LABM2)
            df_TOTFAE = query_loop(df_TOTFAE)
            df_TOTFAO1 = query_loop(df_TOTFAO1)
            df_TOTFAO2 = query_loop(df_TOTFAO2)
            df_TOTFBE = query_loop(df_TOTFBE)
            df_TOTFBO1 = query_loop(df_TOTFBO1)
            df_TOTFBO2 = query_loop(df_TOTFBO2)
        else:
            pass

        def table_count():

            global CASE_count,CRLF_count,CRSF_count,DEATH_count,LABD1_count,LABD2_count,LABM1_count,LABM2_count,TOTFA_count,TOTFB_count,combine_count
            CASE_count = df_CASE.groupby(["id"])
            CASE_count = CASE_count.size().reset_index(name='CASE_total_counts')
            
            CRLF_count = df_CRLF.groupby(["id"])
            CRLF_count = CRLF_count.size().reset_index(name='CRLF_total_counts')
            
            # CRSF_count = df_CRSF.groupby(["id"])
            # CRSF_count = CRSF_count.size().reset_index(name='CRSF_total_counts')
            
            DEATH_count = df_DEATH.groupby(["id"])
            DEATH_count = DEATH_count.size().reset_index(name='DEATH_total_counts')        
            
            # LABD1_count = df_LABD1.groupby(["id"])
            # LABD1_count = LABD1_count.size().reset_index(name='LABD1_total_counts') 
            
            # LABD2_count = df_LABD2.groupby(["id"])
            # LABD2_count = LABD2_count.size().reset_index(name='LABD2_total_counts')
            
            LABM1_count = df_LABM1.groupby(["id"])
            LABM1_count = LABM1_count.size().reset_index(name='LAB1_total_counts')
            
            LABM2_count = df_LABM2.groupby(["id"])
            LABM2_count = LABM2_count.size().reset_index(name='LAB2_total_counts')
            
            TOTFAE_count = df_TOTFAE.groupby(["id"])
            TOTFAE_count = TOTFAE_count.size().reset_index(name='TOTFAE_total_counts')
            
            
            TOTFBE_count = df_TOTFBE.groupby(["id"])
            TOTFBE_count = TOTFBE_count.size().reset_index(name='TOTFBE_total_counts')
            
            TOTFAO1_count = df_TOTFAO1.groupby(["id"])
            TOTFAO1_count = TOTFAO1_count.size().reset_index(name='TOTFAO1_total_counts')

            TOTFAO2_count = df_TOTFAO2.groupby(["id"])
            TOTFAO2_count = TOTFAO2_count.size().reset_index(name='TOTFAO2_total_counts')

            TOTFBO1_count = df_TOTFBO1.groupby(["id"])
            TOTFBO1_count = TOTFBO1_count.size().reset_index(name='TOTFBO1_total_counts')
            
            TOTFBO2_count = df_TOTFBO2.groupby(["id"])
            TOTFBO2_count = TOTFBO2_count.size().reset_index(name='TOTFBO2_total_counts')
        
            combine_count = pd.merge(CASE_count, CRLF_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            # combine_count = pd.merge(CRSF_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(DEATH_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            # combine_count = pd.merge(LABD1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            # combine_count = pd.merge(LABD2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(LABM1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(LABM2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)        
            combine_count = pd.merge(TOTFAE_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(TOTFBE_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(TOTFAO1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(TOTFAO2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(TOTFBO1_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)
            combine_count = pd.merge(TOTFBO2_count, combine_count, how='outer', on=['id'], indicator=False).fillna(value=0)

            return(combine_count)
        
        def Form1_new_table(): 
            global Form1_new_table
            # try:
            #     df_CRLF['age']=df_CRLF['age'].astype(int)
            # except:
            #     pass
            df_CRLF['age_group'] = df_CRLF['age'].apply(age_group)
            Form1_new_table = df_CRLF[['id','sex','resid','dbirth','age','age_group','site','didiag','grade_c','grade_p','cstage','pstage','vstatus','ssf1','ssf2','ssf3','ssf4','ssf5','ssf6','ssf7','ssf8','ssf9','ssf10']]
            Form1_new_table = Form1_new_table.drop_duplicates(subset = ["id","site","didiag"],keep='last')
            Form1_new_table['dbirth'] = Form1_new_table['dbirth'].apply(fourdate)
            Form1_new_table['didiag'] = Form1_new_table['didiag'].apply(fourdate)
            return Form1_new_table

        #-合併CRLF挑的欄位 & 各表的計筆數
        global up,down,Form1_new_table_sult
        up = table_count()
        down = Form1_new_table()
        Form1_new_table_sult = pd.merge(down, up, how='outer', on=['id'], indicator=False)

        Form1_new_table_sult['id'] = Form1_new_table_sult['id'].astype(object)
        Form1_new_table_sult['sex'] = Form1_new_table_sult['sex'].astype(object)
        Form1_new_table_sult['resid'] = Form1_new_table_sult['resid'].astype(object)
        Form1_new_table_sult['dbirth'] = Form1_new_table_sult['dbirth'].astype(object)
        Form1_new_table_sult['age_group'] = Form1_new_table_sult['age_group'].astype(object)
        Form1_new_table_sult['site'] = Form1_new_table_sult['site'].astype(object)
        Form1_new_table_sult['didiag'] = Form1_new_table_sult['didiag'].astype(object)
        Form1_new_table_sult['grade_c'] = Form1_new_table_sult['grade_c'].astype(object)
        Form1_new_table_sult['grade_p'] = Form1_new_table_sult['grade_p'].astype(object)
        Form1_new_table_sult['vstatus'] = Form1_new_table_sult['vstatus'].astype(object)
        Form1_new_table_sult['ssf1'] = Form1_new_table_sult['ssf1'].astype(object)
        Form1_new_table_sult['ssf2'] = Form1_new_table_sult['ssf2'].astype(object)
        Form1_new_table_sult['ssf3'] = Form1_new_table_sult['ssf3'].astype(object)
        Form1_new_table_sult['ssf4'] = Form1_new_table_sult['ssf4'].astype(object)
        Form1_new_table_sult['ssf5'] = Form1_new_table_sult['ssf5'].astype(object)
        Form1_new_table_sult['ssf6'] = Form1_new_table_sult['ssf6'].astype(object)   
        Form1_new_table_sult['ssf7'] = Form1_new_table_sult['ssf7'].astype(object)
        Form1_new_table_sult['ssf8'] = Form1_new_table_sult['ssf8'].astype(object)
        Form1_new_table_sult['ssf9'] = Form1_new_table_sult['ssf9'].astype(object)
        Form1_new_table_sult['ssf10'] = Form1_new_table_sult['ssf10'].astype(object)
        Form1_new_table_sult = Form1_new_table_sult.replace(r'^\s*$',np.nan,regex=True)
        if stats=='update' or len(logic)>= 5:
            pass
        else:
            path = './tmp_json'
            if not os.path.isdir(path):
                os.mkdir(path)
            Form1_new_table_sult = insert_age_gender(Form1_new_table_sult)
            Form1_new_table_sult.to_json("./tmp_json/Summary.json", orient = 'records',date_format = 'iso', double_precision=3)

        des_Form1_new_table_sult = describe(Form1_new_table_sult).T
        des_Form1_new_table_sult  = missing(des_Form1_new_table_sult, Form1_new_table_sult)
        des_Form1_new_table_sult_v  = split_v_c(des_Form1_new_table_sult)[0]
        des_Form1_new_table_sult_v = summ(des_Form1_new_table_sult_v.T,Form1_new_table_sult).T
        des_Form1_new_table_sult_c  = split_v_c(des_Form1_new_table_sult)[1]
        
        conn.close()  
        return(Form1_new_table_sult,des_Form1_new_table_sult_v,des_Form1_new_table_sult_c)
    except:
        Form1_new_table_sult = pd.DataFrame({'no_match':['please update oneclick']}) #預防沒按過一鍵更新前端form1報錯
        
        conn.close()
        return(Form1_new_table_sult,Form1_new_table_sult,Form1_new_table_sult)
 ################################################################################
 ################################################################################

def B_plus_form1_tmp(logic):
    try:
        Summary_df = pd.read_json('./tmp_json/Summary.json', orient ='records')
        Summary_df['id'] = Summary_df['id'].astype(object)
        Summary_df['sex'] = Summary_df['sex'].astype(object)
        Summary_df['resid'] = Summary_df['resid'].astype(object)
        Summary_df['dbirth'] = Summary_df['dbirth'].astype(object)
        Summary_df['age_group'] = Summary_df['age_group'].astype(object)
        Summary_df['site'] = Summary_df['site'].astype(object)
        Summary_df['didiag'] = Summary_df['didiag'].astype(object)
        Summary_df['grade_c'] = Summary_df['grade_c'].astype(object)
        Summary_df['grade_p'] = Summary_df['grade_p'].astype(object)
        Summary_df['vstatus'] = Summary_df['vstatus'].astype(object)
        Summary_df['ssf1'] = Summary_df['ssf1'].astype(object)
        Summary_df['ssf2'] = Summary_df['ssf2'].astype(object)
        Summary_df['ssf3'] = Summary_df['ssf3'].astype(object)
        Summary_df['ssf4'] = Summary_df['ssf4'].astype(object)
        Summary_df['ssf5'] = Summary_df['ssf5'].astype(object)
        Summary_df['ssf6'] = Summary_df['ssf6'].astype(object)   
        Summary_df['ssf7'] = Summary_df['ssf7'].astype(object)
        Summary_df['ssf8'] = Summary_df['ssf8'].astype(object)
        Summary_df['ssf9'] = Summary_df['ssf9'].astype(object)
        Summary_df['ssf10'] = Summary_df['ssf10'].astype(object)   

        if len(logic)>3:

            logic_query = logic_json(logic)

            if len(logic_query)<3:
                return(ex_df,ex_df,ex_df)

            match_df = pd.DataFrame()
            query_batch = logic_query.split('|')
            print(len(query_batch))
            i=0
            for b in range(len(query_batch)):
                df = Summary_df.query(query_batch[i])
                match_df = pd.concat([match_df, df], axis=0)
                i=i+1

            des_match_df = describe(match_df).T
            des_match_df  = missing(des_match_df, match_df)
            des_match_df_v  = split_v_c(des_match_df)[0]
            des_match_df_v = summ(des_match_df_v.T,match_df).T
            des_match_df_c  = split_v_c(des_match_df)[1]

            return(match_df.fillna('-'),des_match_df_v,des_match_df_c)

        else:
            match_df = Summary_df
            des_match_df = describe(match_df).T
            des_match_df  = missing(des_match_df, match_df)
            des_match_df_v  = split_v_c(des_match_df)[0]
            des_match_df_v = summ(des_match_df_v.T,match_df).T
            des_match_df_c  = split_v_c(des_match_df)[1]

            return(match_df.fillna('-'),des_match_df_v,des_match_df_c)
    except:
        Summary_df = pd.DataFrame({'no_match':['please update oneclick']}) #預防沒按過一鍵更新前端form1報錯
        return(Summary_df,Summary_df,Summary_df)

def table_info(table,stats,logic):
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)
    cursor = conn.cursor()

    # today_date = "'"+time.strftime("%Y-%m-%d 00:00:00")+"'"

    table_name = str(table)
    df = pd.read_sql("SELECT Max(ModifyTime) FROM " + "[" + table_name + "]" , conn)
    df['Max(ModifyTime)']= pd.to_datetime(df['Max(ModifyTime)']) - pd.Timedelta(days=1)
    df = df.astype(str)
    today_date = "'"+df.iat[0,0]+"'"    

    if len(logic)>3:

        logic_query = logic_json(logic)

        if len(logic_query)<3:
            return(ex_df,ex_df,ex_df)

    #sql to pandas
    global dataframe_info,df_right
    if stats =='update':
        dataframe_info  = pd.read_sql("SELECT * FROM " + "[" + table + "]" + "where ModifyTime>=" + today_date, conn)
        if dataframe_info.empty==True:
            return(ex_df,ex_df,ex_df)
        try:
            dataframe_info.rename(columns={'d3':'id'}, inplace = True)
        except:
            pass
        try:
            dataframe_info.rename(columns={'h9':'id'}, inplace = True)
        except:
            pass


        if len(logic) >= 2:
            # logic_query = logic_json(logic) #一開始為了判斷篩不到人跑過了，不二跑
            match_df = pd.DataFrame()
            query_batch = logic_query.split('|')

            i=0
            for b in range(len(query_batch)):
                df = dataframe_info.query(query_batch[i])
                match_df = pd.concat([match_df, df], axis=0)
                i=i+1

            dataframe_info = match_df

            # dataframe_info = dataframe_info.query(logic_query)
        else:
            pass

        dataframe_info_orig = dataframe_info.copy()

        try:
            dataframe_info = dataframe_info.drop_duplicates(subset = ["id"])
        except:
            pass

    else:
        dataframe_info  = pd.read_sql("SELECT * FROM " + "[" + table + "]", conn)
        if dataframe_info.empty==True:
            return(ex_df,ex_df,ex_df)
        try:
            dataframe_info.rename(columns={'d3':'id'}, inplace = True)
        except:
            pass

        try:
            dataframe_info.rename(columns={'h9':'id'}, inplace = True)
        except:
            pass

        if len(logic) >= 2:
            # logic_query = logic_json(logic) #一開始為了判斷篩不到人跑過了，不二跑
            match_df = pd.DataFrame()
            query_batch = logic_query.split('|')

            i=0
            for b in range(len(query_batch)):
                df = dataframe_info.query(query_batch[i])
                match_df = pd.concat([match_df, df], axis=0)
                i=i+1

            dataframe_info = match_df

            # dataframe_info = dataframe_info.query(logic_query) #一開始為了判斷篩不到人跑過了，不二跑
        else:
            pass

        dataframe_info_orig = dataframe_info.copy()

        try:
            dataframe_info = dataframe_info.drop_duplicates(subset = ["id"])
        except:
            pass       

    try:
        c = pd.crosstab(dataframe_info['gender'], dataframe_info['gender'], margins=True, margins_name='total') #[0]欄位類別 [1]統計變數
        df_right = c.T
        df_right = df_right.tail(1) #顯示最後總合的就好
        df_right.rename(columns={'1':'male', '2':'female'}, inplace = True)
        if 'male' in df_right.columns:
            pass
        else:
            df_right['male']=0

        if 'female' in df_right.columns:
            pass
        else:
            df_right['female']=0
    except:
        pass
    try:
        c = pd.crosstab(dataframe_info['sex'], dataframe_info['sex'], margins=True, margins_name='total') #[0]欄位類別 [1]統計變數
        df_right = c.T
        df_right = df_right.tail(1) #顯示最後總合的就好
        df_right.rename(columns={'1':'male', '2':'female'}, inplace = True)
        if 'male' in df_right.columns:
            pass
        else:
            df_right['male']=0

        if 'female' in df_right.columns:
            pass
        else:
            df_right['female']=0
    except:
        pass

    row = str(len(dataframe_info_orig.index))
    row = {"table_line": row }
    row = pd.DataFrame([row])
    Newest_date = dataframe_info['CreateTime'].max()
    Oldest_date = dataframe_info['CreateTime'].min()
    range_date = {"Newest_date": Newest_date ,"Oldest_date":Oldest_date }
    range_date = pd.DataFrame([range_date])
    
    conn.close()
    return(df_right, row, range_date)

def One_click_update(last_update):

    path = './tmp_json'
    if not os.path.isdir(path):
        os.mkdir(path)

    conn = sqlite3.connect('C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db')
    cursor = conn.cursor()

    df_CASE = pd.read_sql("SELECT id FROM " + "[" + 'CASE' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    CASE_ID = df_CASE['id'].drop_duplicates().tolist()

    df_CRLF = pd.read_sql("SELECT id FROM " + "[" + 'CRLF' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    CRLF_ID = df_CRLF['id'].drop_duplicates().tolist()

    # df_CRSF = pd.read_sql("SELECT id FROM " + "[" + 'CRSF' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    # CRSF_ID = df_CRSF['id'].drop_duplicates().tolist()

    df_DEATH = pd.read_sql("SELECT id FROM " + "[" + 'DEATH' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    DEATH_ID = df_DEATH['id'].drop_duplicates().tolist()

    # df_LABD1 = pd.read_sql("SELECT h9 FROM " + "[" + 'LABD1' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    # LABD1_ID = df_LABD1['h9'].drop_duplicates().tolist()

    # df_LABD2 = pd.read_sql("SELECT h9 FROM " + "[" + 'LABD2' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    # LABD2_ID = df_LABD2['h9'].drop_duplicates().tolist()

    df_LABM1 = pd.read_sql("SELECT h9 FROM " + "[" + 'LABM1' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    LABM1_ID = df_LABM1['h9'].drop_duplicates().tolist()

    df_LABM2 =  pd.read_sql("SELECT h9 FROM " + "[" + 'LABM2' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    LABM2_ID = df_LABM2['h9'].drop_duplicates().tolist()

    df_TOTFAE = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFAE' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    TOTFAE_ID = df_TOTFAE['d3'].drop_duplicates().tolist()

    df_TOTFAO1 = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFAO1' + "]" + "where ModifyTime>" + "'" + last_update + "'"  + "AND d3 is not null" , conn)
    TOTFAO1_ID = df_TOTFAO1['d3'].drop_duplicates().tolist()

    df_TOTFAO2 = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFAO2' + "]" + "where ModifyTime>" + "'" + last_update + "'"  + "AND d3 is not null" , conn)
    TOTFAO2_ID = df_TOTFAO2['d3'].drop_duplicates().tolist()

    df_TOTFBE = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFBE' + "]" + "where ModifyTime>" + "'" + last_update + "'" , conn)
    TOTFBE_ID = df_TOTFBE['d3'].drop_duplicates().tolist()

    df_TOTFBO1 = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFBO1' + "]" + "where ModifyTime>" + "'" + last_update + "'"  + "AND d3 is not null" , conn)
    TOTFBO1_ID = df_TOTFBO1['d3'].drop_duplicates().tolist()

    df_TOTFBO2 = pd.read_sql("SELECT d3 FROM " + "[" + 'TOTFBO2' + "]" + "where ModifyTime>" + "'" + last_update + "'"  + "AND d3 is not null" , conn)
    TOTFBO2_ID = df_TOTFBO2['d3'].drop_duplicates().tolist()

    SET_ID = set(CASE_ID) | set(CRLF_ID)  | set(DEATH_ID) | set(TOTFAO2_ID) | set(LABM1_ID) | set(LABM2_ID) | set(TOTFAE_ID) | set(TOTFAO1_ID) | set(TOTFAO2_ID) | set(TOTFBE_ID) | set(TOTFBO1_ID) | set(TOTFBO2_ID)
    SET_ID = list(SET_ID)

    def finish_percent():
        finish_percent = cursor.execute("""SELECT 100*(SELECT count(*) from LOG WHERE state==1 ) / (SELECT count(*) FROM LOG WHERE state=1 or state=0) as total """)
        for row in cursor:
            finish_percent = row[0]
            finish_percent = str(row[0])
        return(finish_percent)

    def finish_time():
        finish_time = cursor.execute("""SELECT max(finish_time) FROM LOG as max_time""")
        for row2 in cursor:
            finish_time = str(row2[0])
        return(finish_time)

    if not SET_ID: #如果mod-id沒人了
        print('not found mod-id')
        df_log = pd.read_sql("SELECT id FROM " + "[" + 'LOG' + "]" + "where state=0" , conn)
        if df_log.empty ==True: #如果state=0沒人了
            finish_percent = finish_percent()
            finish_time = finish_time()
            print('not found state0')
            return("not found mod-id &  not found state0"+","+str(finish_percent)+","+str(finish_time)) #都沒找到
        else: #state=0有人, 算0的人不用1改0再改1
            print('find state0')
            into_ID = df_log['id'].drop_duplicates().tolist()
            print(into_ID)

        def func(ID):
            form2_back_sult = B_plus_form2(ID,'')
            form2_back_sult.to_json("./tmp_json/"+ID+".json", orient = 'records',date_format = 'iso', double_precision=3)
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S")
            sql_update_query = """Update LOG set state = 1,finish_time =""" +"'"+ finish_time+"'" + """where id =""" +"'"+ID+"'"
            cursor.execute(sql_update_query)
            conn.commit()
        
        j=0
        for j in range(len(into_ID)):
            try:
                func(into_ID[j])
                j+=1
            except Exception as e:
                error_class = e.__class__.__name__ #取得錯誤類型
                detail = e.args[0] #取得詳細內容
                cl, exc, tb = sys.exc_info() #取得Call Stack
                lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
                fileName = lastCallStack[0] #取得發生的檔案名稱
                lineNum = lastCallStack[1] #取得發生的行號
                funcName = lastCallStack[2] #取得發生的函數名稱
                errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
                print(errMsg)
                f=open("error.txt","a")
                f.write(errMsg+today_date)
                f.write('\n')
                f.close()
                j+=1
        finish_percent = finish_percent()
        finish_time = finish_time()
        return("not found mod-id & finish state0"+","+str(finish_percent)+","+str(finish_time)) #沒找到要更新的,但找到0的

    else: #mod-id有人
        print('find mod-id')
        LOG_df_one = pd.DataFrame(SET_ID, columns = ['id'])
        LOG_df_one['state'] = 0 #把要更新的人都改為0
        LOG_df_one['event_time'] = time.strftime("%Y-%m-%d %H:%M:%S") #str
        LOG_df_one['event_time'] = LOG_df_one['event_time'].astype(str)

        df_col = list(LOG_df_one.itertuples(index=False, name=None)) #df轉tuple

        k=0
        for k in range(len(df_col)):
            
            sqlite_upsert_query = """INSERT INTO LOG
                                      (id, state, event_time)
                                       VALUES 
                                      (?,?,?)
                                      ON CONFLICT(id) DO UPDATE SET 
                                      state=excluded.state,
                                      event_time=excluded.event_time
                                      """

            count = cursor.execute(sqlite_upsert_query,df_col[k])
            conn.commit()
            k+=1

        df_log = pd.read_sql("SELECT id FROM " + "[" + 'LOG' + "]" + "where state=0" , conn)
        into_ID = df_log['id'].drop_duplicates().tolist()

        def func(ID):
            form2_back_sult = B_plus_form2(ID,'')
            form2_back_sult.to_json("./tmp_json/"+ID+".json", orient = 'records',date_format = 'iso', double_precision=3)
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S")
            sql_update_query = """Update LOG set state = 1,finish_time =""" +"'"+ finish_time+"'" + """where id =""" +"'"+ID+"'"
            cursor.execute(sql_update_query)
            conn.commit()
        
        j=0
        for j in range(len(into_ID)):
            try:               
                func(into_ID[j])
                j+=1
            except Exception as e:
                error_class = e.__class__.__name__ #取得錯誤類型
                detail = e.args[0] #取得詳細內容
                cl, exc, tb = sys.exc_info() #取得Call Stack
                lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
                fileName = lastCallStack[0] #取得發生的檔案名稱
                lineNum = lastCallStack[1] #取得發生的行號
                funcName = lastCallStack[2] #取得發生的函數名稱
                errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
                print(errMsg)
                f=open("error.txt","a")
                f.write(errMsg+today_date)
                f.write('\n')
                f.close()
                j+=1

        finish_percent = finish_percent()
        finish_time = finish_time()
        
        conn.close()
        return("FINISH"+","+str(finish_percent)+","+str(finish_time))

def progress():
    conn = sqlite3.connect('C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db')
    cursor = conn.cursor()

    def progress_percent():
        percent = cursor.execute("""SELECT 100*(SELECT count(*) from LOG WHERE state==1 ) / (SELECT count(*) FROM LOG WHERE state=1 or state=0) as total """)
        for row in cursor:
            percent = row[0]
            percent = str(row[0])
        return(percent)

    def progress_time():
        time = cursor.execute("""SELECT max(finish_time) FROM LOG as max_time""")
        for row2 in cursor:
            time = str(row2[0])
        return(time)
    percent = progress_percent()
    time = progress_time()
    
    conn.close()
    return(percent+','+time)

def detail_value_count(col):

    def full0(x):
        if len(x)<3:
            x = x.zfill(3)
        return x

    df = pd.read_json('./tmp_json/Summary.json', orient ='records')
    df = df[[col]]
    freq = df.groupby([col]).count()
    freq1 = df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
    freq2 = df[col].value_counts()
    freq1=freq1.to_frame()
    freq2=freq2.to_frame()
    freq1.rename(columns = {col:col+'_1'}, inplace = True)
    freq2.rename(columns = {col:col+'_2'},inplace = True)
    res = pd.concat([freq1,freq2],axis=1)
    res = res.astype(str)
    res[col] = res[col+'_2'] +" "+ "("+  res[col+'_1'] + ")"
    res = res[[col]]
    res['type'] = res.index
    res.rename(columns = {col:"value_count"},inplace = True)
    res = pd.concat([res,freq2],axis=1)
    res.rename(columns = {col+'_2':"value"},inplace = True) 
    res = res[['type','value_count','value']]
    res = res.astype(str)
    res['type'] = res['type'].apply(full0)
    return(res)

def C_CANCER(logic_structure):

    #DB
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor
    #定位
    sort = logic_structure.get("table_select")
    i=0
    global table_list
    table_list = []
    for i in range(len(sort)):
        table_list.append(sort[i]['table'])
        i=i+1   
     #common function
    def cut_date(x):
        try:          
            if len(x)>8:
                x = x[0:8]
            x_f = x[0:4]
            x_f = x_f.replace('9999','1900')
            x_m = x[4:6]
            x_m = x_m.replace('99','01')
            x_b = x[6:8]
            x_b = x_b.replace('99','01')
            x = x_f + x_m + x_b
            x = parse(x)
            return x
        
        except:
            x_error = '19000101'
            x_error = parse(x_error)
            return x_error

    #cohrot 重要項 
    disease = logic_structure['disease']
    keep = logic_structure['keep']
    search_id = logic_structure['search_id']

    #write index path
    index_write = str(logic_structure['index_write'])
    is_coop = str(logic_structure['coop'])
    if is_coop == "0":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer\\" + index_write +"\\config\\"
    if is_coop == "1":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop\\" + index_write +"\\config\\"
    ###0%進度點###
    process_path = plug_path+'C_process.txt'
    try:
        os.remove(process_path)
    except:
        pass
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('0%')
    f.close()
    ###0%進度點###

    # CRLF ############################篩選結果資料(大excel用)、帶時間ID冊(日期用、可能與各表交集用)
    #astype and site query
    CRLF_index = table_list.index('CRLF')
    # df_CRLF = pd.read_sql("SELECT id,sex,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM " + "CRLF", conn)
    df_CRLF = pd.read_sql("SELECT * FROM " + "CRLF", conn)
    df_CRLF = df_CRLF.astype(str)
    df_CRLF['age'] = df_CRLF['age'].astype(int)
    df_CRLF = df_CRLF.query(search_id,engine='python')
    df_CRLF = df_CRLF.query(disease,engine='python')
    if df_CRLF.empty == True:
        ###離開100%進度點###
        process_path = plug_path+'C_process.txt'
        f = open(process_path, 'w')
        f.write('100%')
        f.close()
        ###離開100%進度點###
        nomatch_df = pd.DataFrame({'no_match':['CRLF Site code no matching results']})
        return(nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df)
    
    #queryList(複數個欄位):篩選結果資料
    i=0
    for each_query in range(len(logic_structure['table_select'][CRLF_index]['queryList'])):
        each = list(logic_structure['table_select'][CRLF_index]['queryList'].values())[i][0]
        df_CRLF = df_CRLF.query(each)
        i+=1
    
    if df_CRLF.empty == True:
        nomatch_df = pd.DataFrame({'no_match':['CRLF Columns condition code no matching results']})
        ###離開100%進度點###
        process_path = plug_path+'C_process.txt'
        f = open(process_path, 'w')
        f.write('100%')
        f.close()
        ###離開100%進度點###
        return(nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df)
    
    df_CRLF #CRLF結果
    
    #cohort_time:帶時間ID冊
    df_CRLF_cohort = df_CRLF[['id','didiag']]
    df_CRLF_cohort['didiag'] = df_CRLF_cohort['didiag'].apply(cut_date)
    df_CRLF_cohort

    # #存活ID冊
    # df_CRLF_ID = df_CRLF_cohort['id'].drop_duplicates().tolist()
    # df_CRLF_ID
 
    ###10%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('10%')
    f.close()
    ###10%進度點###

 ######################## TOTFAE ########################## 正常篩選結果、無效回報

    #function
    def TOTFAE_count(Col_Name, ICD):
            _19 = df_TOTFAE.loc[df_TOTFAE['d19'].str.startswith((ICD), na = False)] #各欄找ICD
            _20 = df_TOTFAE.loc[df_TOTFAE['d20'].str.startswith((ICD), na = False)]
            _21 = df_TOTFAE.loc[df_TOTFAE['d21'].str.startswith((ICD), na = False)]
            _22 = df_TOTFAE.loc[df_TOTFAE['d22'].str.startswith((ICD), na = False)]
            _23 = df_TOTFAE.loc[df_TOTFAE['d23'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_19,_20,_21,_22,_23])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            AE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            AE_row_count = AE_row_count.to_frame()
            AE_row_count = AE_row_count.reset_index()
            AE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return AE_row_count
        
    #comobidity
    TOTFAE_index = table_list.index('TOTFAE')
    comobidity_num = len(logic_structure['table_select'][TOTFAE_index]['logic_before'])
    if comobidity_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFAE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        AE_row_count_before = pd.DataFrame({'id':['N/A'],'comobidity_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        # df_TOTFAE = pd.read_sql("SELECT * FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_CRLF_cohort, df_TOTFAE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFAE.empty == True:
            AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE內 無同癌登的人群']})
        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['didiag'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600
                
            #before
            df_TOTFAE_before = df_TOTFAE.query("diff_days <=" + str(start_time_TOTFAE) + " " + "and" + " " + "diff_days >=" + str(0))
            
            if df_TOTFAE_before.empty == True:
                AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity time no matching results']})
    
            else:
                comobidity_col_name = []    #收集config中的所有comobidity_name
                comobidity_condition =[]    #收集config中的所有comobidity_condition

                for c in range(comobidity_num):
                    comobidity_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['col_name'])
                    comobidity_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['condition'])

                c_ = 0
                for c_ in range(comobidity_num):
                    if c_ == 0:
                        AE_row_count_before = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                    else:
                        AE_row_count_before_append = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                        AE_row_count_before = pd.merge(AE_row_count_before, AE_row_count_before_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d=0
                # for d in range(len(comobidity_col_name)):
                #     AE_row_count_before = AE_row_count_before.drop(AE_row_count_before[AE_row_count_before[comobidity_col_name[d]]<2].index) #刪除門診<2的人

                if AE_row_count_before.empty == True:
                    AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity count no matching results with >=2']})
                    
                else:
                    TOTFAE_ID_before = AE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    AE_row_count_before #輸出結果表
                    
    #complication
    TOTFAE_index = table_list.index('TOTFAE')
    complication_num = len(logic_structure['table_select'][TOTFAE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFAE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        AE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_CRLF_cohort, df_TOTFAE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFAE.empty == True:
            AE_row_count_after  = pd.DataFrame({'no_match':['TOTFAE內 無同癌登的人群']})
            
        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['didiag'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600
                
            #after
            df_TOTFAE_after = df_TOTFAE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAE))
            
            if df_TOTFAE_after.empty == True:
                AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication time no matching results']})
    
            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition
                
                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['condition'])
       
                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        AE_row_count_after = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        AE_row_count_after_append = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        AE_row_count_after = pd.merge(AE_row_count_after, AE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d = 0
                # for d in range(len(complication_col_name)):
                #     AE_row_count_after = AE_row_count_after.drop(AE_row_count_after[AE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人
                    
                if AE_row_count_after.empty == True:
                    AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication count no matching results with >=2']})
                    
                else:
                    TOTFAE_ID_after = AE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    AE_row_count_after #輸出結果表

    ###20%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('20%')
    f.close()
    ###20%進度點###

 ######################## TOTFBE ########################## 正常篩選結果、無效回報
    #function
    def TOTFBE_count(Col_Name, ICD):
            _25 = df_TOTFBE.loc[df_TOTFBE['d25'].str.startswith((ICD), na = False)] #各欄找ICD
            _26 = df_TOTFBE.loc[df_TOTFBE['d26'].str.startswith((ICD), na = False)]
            _27 = df_TOTFBE.loc[df_TOTFBE['d27'].str.startswith((ICD), na = False)]
            _28 = df_TOTFBE.loc[df_TOTFBE['d28'].str.startswith((ICD), na = False)]
            _29 = df_TOTFBE.loc[df_TOTFBE['d29'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_25,_26,_27,_28,_29])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            BE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            BE_row_count = BE_row_count.to_frame()
            BE_row_count = BE_row_count.reset_index()
            BE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return BE_row_count
        
    #comobidity
    TOTFBE_index = table_list.index('TOTFBE')
    comobidity_num = len(logic_structure['table_select'][TOTFBE_index]['logic_before'])
    
    if comobidity_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        BE_row_count_before = pd.DataFrame({'id':['N/A'],'comorbidity_N/A':['N/A']}) 
        
    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_CRLF_cohort, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date)
        
        if df_TOTFBE.empty == True:
            BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
            BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
        else:
            
            df_TOTFBE['diff_days'] = df_TOTFBE['didiag'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600
            
            #before
            df_TOTFBE_before = df_TOTFBE.query("diff_days <=" + str(start_time_TOTFBE) + " " + "and" + " " + "diff_days >=" + str(0))
            
            if df_TOTFBE_before.empty == True:
                BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity time no matching results']})
            else:
                comobidity_col_name = []    #收集config中的所有comobidity_name
                comobidity_condition =[]    #收集config中的所有comobidity_condition
                
                for c in range(comobidity_num):
                    comobidity_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['col_name'])
                    comobidity_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['condition'])
                c_ = 0
                for c_ in range(comobidity_num):
                    if c_ == 0:
                        BE_row_count_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_append_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                        BE_row_count_before = pd.merge(BE_row_count_before, BE_row_count_append_before, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1

                # d = 0
                # for d in range(len(comobidity_col_name)):
                #     BE_row_count_before = BE_row_count_before.drop(BE_row_count_before[BE_row_count_before[comobidity_col_name[d]]<1].index) #刪除住院<1的人

                if BE_row_count_before.empty == True:
                    BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity count no matching results with >=1']})
                    
                else:
                    TOTFBE_ID_before = BE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_before #輸出結果表
                    
    #complication
    TOTFBE_index = table_list.index('TOTFBE')
    complication_num = len(logic_structure['table_select'][TOTFBE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        BE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_CRLF_cohort, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFBE.empty == True:
            BE_row_count_after  = pd.DataFrame({'no_match':['TOTFBE內 無同癌登的人群']})
            
        else:
            df_TOTFBE['diff_days'] = df_TOTFBE['didiag'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600
                
            #after
            df_TOTFBE_after = df_TOTFBE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBE))
            
            if df_TOTFBE_after.empty == True:
                BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication time no matching results']})
    
            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition

                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['condition'])
                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        BE_row_count_after = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_after_append = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        BE_row_count_after = pd.merge(BE_row_count_after, BE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d = 0
                # for d in range(len(complication_col_name)):
                #     BE_row_count_after = BE_row_count_after.drop(BE_row_count_after[BE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人
                    
                if BE_row_count_after.empty == True:
                    BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication count no matching results with >=2']})
                    
                else:
                    TOTFBE_ID_after = BE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_after #輸出結果表

    ###30%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('30%')
    f.close()
    ###30%進度點###

    # TOTFA01 ######################### 篩選結果存活ID冊、藥品細項清單                
    TOTFAO1_index = table_list.index('TOTFAO1')
     # df_TOTFAO1 = pd.read_sql("SELECT d3,p4,d9 FROM " + "TOTFAO1", conn)
    df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "TOTFAO1", conn)
    df_TOTFAO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFAO1 = pd.merge(df_CRLF_cohort, df_TOTFAO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFAO1['d9'] = df_TOTFAO1['d9'].apply(cut_date)

    if df_TOTFAO1.empty == True:
        TOTFAO1_ID = df_CRLF[['id']]
        TOTFAO1_match_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p4':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p9':['N/A'],'p10':['N/A'],'p13':['N/A'],'p14':['N/A'],'p15':['N/A'],'p17':['N/A'],'d9':['N/A']})
    else:
        df_TOTFAO1['diff_days'] = df_TOTFAO1['didiag'] - df_TOTFAO1['d9']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFAO1['diff_days'] = df_TOTFAO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFAO1 = int(logic_structure['table_select'][TOTFAO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFAO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFAO1 = -(int(logic_structure['table_select'][TOTFAO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFAO1 = -3600

        df_TOTFAO1 = df_TOTFAO1.query("diff_days <=" + str(start_time_TOTFAO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAO1))#動時間還未進藥篩
        
        if df_TOTFAO1.empty == True:
            TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug time no matching results']})
            
        else:
            TOTFAO1_match_df = pd.DataFrame()
            query_TOTFAO1 = logic_structure['table_select'][TOTFAO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFAO1_batch = query_TOTFAO1.split('|')

            try: #防空表
                i=0
                for A in range(len(query_TOTFAO1_batch)):
                    insert_df = df_TOTFAO1.query(query_TOTFAO1_batch[i])
                    TOTFAO1_match_df = pd.concat([TOTFAO1_match_df, insert_df], axis=0)
                    TOTFAO1_match_df = TOTFAO1_match_df.drop_duplicates()
                    i+=1

                if TOTFAO1_match_df.empty == True:
                    TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})
                    
                else:
                    #藥品細項清單
                    TOTFAO1_match_df

                    #存活ID冊
                    TOTFAO1_ID = TOTFAO1_match_df['id'].drop_duplicates().tolist()
            except:
                TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})
                TOTFAO1_ID=[]

    ###40%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('40%')
    f.close()
    ###40%進度點###

    # TOTFB01 ######################### 篩選結果存活ID冊、醫令細項清單
    df_TOTFBO1_sql = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)            
    #藥物
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchD_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchD_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchD_df = pd.concat([TOTFBO1_matchD_df, insert_df], axis=0)
                    TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchD_df.empty == True:
                    TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     
                else:
                    #醫令細項清單
                    TOTFBO1_matchD_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_D = TOTFBO1_matchD_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_D
            except:
                TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_D=[]
                
    #手術
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchS_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_surgery'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_surgery'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchS_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_surgery']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchS_df = pd.concat([TOTFBO1_matchS_df, insert_df], axis=0)
                    TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchS_df.empty == True:
                    TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     

                else:
                    #醫令細項清單
                    TOTFBO1_matchS_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_S = TOTFBO1_matchS_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_S
            except:
                TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_S=[]

    #檢驗檢查
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchC_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_check'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_check'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchC_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_check']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try: 
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchC_df = pd.concat([TOTFBO1_matchC_df, insert_df], axis=0)
                    TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchC_df.empty == True:
                    TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     

                else:
                    #醫令細項清單
                    TOTFBO1_matchC_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_C = TOTFBO1_matchC_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_C
            except:
                TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_C=[]

    ###50%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('50%')
    f.close()
    ###50%進度點###

    # LABM1_Check ########################### 篩選結果存活ID冊、檢驗細項清單
    LABM1_Check_index = table_list.index('LABM1_Check')
    df_LABM1_Check = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11'] + df_LABM1_Check['h13']
    df_LABM1_Check.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Check = pd.merge(df_CRLF_cohort, df_LABM1_Check, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11|h13'].apply(cut_date)

    if df_LABM1_Check.empty == True:
        LABM1_Check_ID = df_CRLF[['id']]
        LABM1_Check_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'Index':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    
    else:
        df_LABM1_Check['diff_days'] = df_LABM1_Check['didiag'] - df_LABM1_Check['h11|h13']  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Check['diff_days'] = df_LABM1_Check['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Check = int(logic_structure['table_select'][LABM1_Check_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Check = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Check = -(int(logic_structure['table_select'][LABM1_Check_index]['end_time'])*30)
        except:
            end_time_LABM1_Check = -3600

        df_LABM1_Check = df_LABM1_Check.query("diff_days <=" + str(start_time_LABM1_Check) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Check))
        
        if df_LABM1_Check.empty == True:
            LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check time no matching results']})
        
        else:
            LABM1_Check_match_df = pd.DataFrame()
            query_LABM1_Check = logic_structure['table_select'][LABM1_Check_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Check_batch = query_LABM1_Check.split('|')

            try:

                i=0
                for LC in range(len(query_LABM1_Check_batch)):
                    insert_df = df_LABM1_Check.query(query_LABM1_Check_batch[i])
                    LABM1_Check_match_df = pd.concat([LABM1_Check_match_df, insert_df], axis=0)
                    LABM1_Check_match_df = LABM1_Check_match_df.drop_duplicates()
                    i+=1
                    
                if LABM1_Check_match_df.empty == True:
                    LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})

                else:
                    #檢驗細項清單
                    LABM1_Check_match_df
        
                    #存活ID冊
                    LABM1_Check_ID = LABM1_Check_match_df['id'].drop_duplicates().tolist()
                    LABM1_Check_ID
            except:
                LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})
                LABM1_Check_ID=[]
    # LABM1_Surgery ########################### 篩選結果存活ID冊、手術細項清單
    LABM1_Surgery_index = table_list.index('LABM1_Surgery')
    df_LABM1_Surgery = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11'] + df_LABM1_Surgery['h13']
    df_LABM1_Surgery.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Surgery = pd.merge(df_CRLF_cohort, df_LABM1_Surgery, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11|h13'].apply(cut_date)

    if df_LABM1_Surgery.empty == True :
        LABM1_Surgery_ID = df_CRLF[['id']]
        LABM1_Surgery_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'Index':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    else:
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['didiag'] - df_LABM1_Surgery['h11|h13']                                  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Surgery = int(logic_structure['table_select'][LABM1_Surgery_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Surgery = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Surgery = -(int(logic_structure['table_select'][LABM1_Surgery_index]['end_time'])*30)
        except:
            end_time_LABM1_Surgery = -3600

        df_LABM1_Surgery = df_LABM1_Surgery.query("diff_days <=" + str(start_time_LABM1_Surgery) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Surgery))
        
        if df_LABM1_Surgery.empty == True:
            LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery time no matching results']})
        
        else:
            
            LABM1_Surgery_match_df = pd.DataFrame()
            query_LABM1_Surgery = logic_structure['table_select'][LABM1_Surgery_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Surgery_batch = query_LABM1_Surgery.split('|')

            try:

                i=0
                for LS in range(len(query_LABM1_Surgery_batch)):
                    insert_df = df_LABM1_Surgery.query(query_LABM1_Surgery_batch[i])
                    LABM1_Surgery_match_df = pd.concat([LABM1_Surgery_match_df, insert_df], axis=0)
                    LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop_duplicates()
                    i+=1
                
                if LABM1_Surgery_match_df.empty == True:
                    LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})
     
                else:
                    
                    #手術細項清單
                    LABM1_Surgery_match_df

                    #存活ID冊
                    LABM1_Surgery_ID = LABM1_Surgery_match_df['id'].drop_duplicates().tolist()
                    LABM1_Surgery_ID
            except:
                LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})
                LABM1_Surgery_ID=[]

    ###60%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('60%')
    f.close()
    ###60%進度點###

    # CASE ############################ 篩選結果存活ID冊、結果檔            
    CASE_index = table_list.index('CASE')
    df_CASE = pd.read_sql("SELECT * FROM " + "[" + "CASE" + "]", conn)
    df_CASE = pd.merge(df_CRLF_cohort, df_CASE, how='left', on=['id'], indicator=False)
    
    if df_CASE.empty == True :
        df_CASE_ID = df_CRLF[['id']]
        df_CASE_match_df = pd.DataFrame({'Index':['N/A'],'id':['N/A'],'gender':['N/A'],'m2':['N/A'],'m3':['N/A'],'m4':['N/A'],'m5':['N/A'],'m6':['N/A'],'m7':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    
    else:
    
        i=0
        for each_query in range(len(logic_structure['table_select'][CASE_index]['queryList'])):
            each = list(logic_structure['table_select'][CASE_index]['queryList'].values())[i][0]
            df_CASE_match_df = df_CASE.query(each)
            i+=1
        
        if df_CASE_match_df.empty == True:
            df_CASE_match_df = pd.DataFrame({'no_match':['CASE Columns condition code no matching results']})
        
        else:
            #CASE結果
            df_CASE_match_df = df_CASE_match_df.drop(['d3'],axis=1)
            df_CASE_match_df

            #存活ID冊
            df_CASE_ID = df_CASE_match_df['id'].drop_duplicates().tolist()
            df_CASE_ID

    # normal-output ############################        
    try:
        df_CRLF = df_CRLF[keep]
    except:
        df_CRLF = df_CRLF
        
    try:
        AE_row_count_before = AE_row_count_before
    except:
        AE_row_count_before = AE_row_count_before
    
    try:
        AE_row_count_after = AE_row_count_after     
    except:
        AE_row_count_after = AE_row_count_after
    
    try:
        BE_row_count_before = BE_row_count_before   
    except:
        BE_row_count_before = BE_row_count_before

    try:
        BE_row_count_after = BE_row_count_after   
    except:
        BE_row_count_after = BE_row_count_after

    try:
        TOTFAO1_match_df = TOTFAO1_match_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFAO1_match_df = TOTFAO1_match_df
    
    try:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df
    
    try:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df
    
    try:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df
                
    try:
        LABM1_Check_match_df = LABM1_Check_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Check_match_df = LABM1_Check_match_df
        
    try:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df
        
    try:
        df_CASE_match_df = df_CASE_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1)
    except:
        df_CASE_match_df = df_CASE_match_df
    
    ###70%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('70%')
    f.close()
    ###70%進度點###
    
 # HUANG ###############################黃醫師
    # try:
    #     def r2_to_command(x):                            
    #         x = "r2 == " + '\"' + x + '\"'
    #         return(x)

    #     # ALL_LAB = pd.read_sql("SELECT h9,h18,r2,r4,r5,h11,h13 FROM " + "LABM1", conn)
    #     # ALL_LAB['h11|h13'] = ALL_LAB['h11'] + ALL_LAB['h13']
    #     # CRLF = pd.read_sql("SELECT id,gender,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM"  + " CRLF", conn)
    #     ALL_LAB = LABM1_Check_match_df
    #     CRLF = pd.read_sql("SELECT id,sex,age,hist,behavior,smoking,btchew,drinking,ct,cn,cm,cstage,pt,pn,pm,pstage,size,grade_c,grade_p,lateral,site,lvi,pni,smargin,smargin_d,srs,sls,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM " + "CRLF", conn)
    #     # CRLF = pd.read_sql("SELECT id,gender,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM"  + " CRLF", conn)
        
    #     #CEA(ng/ml)
    #     Group_1 = ALL_LAB.query(" h18 == \"12021C\" | h18 == \"27050C\"") #query order code
    #     Group_1['r2'] = Group_1['r2'] + "||" + " (" + Group_1['r5'] + ")"
    #     Group_1 = Group_1.set_index(keys='id',drop=False)         #選定index
    #     Group_1 = pd.get_dummies(Group_1.r2)                      #one hot
    #     Group_1_list = list(Group_1)                              #取出r2 onehot包含哪些項
    #     Group_1_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_1_list)):
    #         insert_df1 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_1_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_1_list[i].replace('||','')
    #         insert_df1.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df1 = insert_df1[['id',column_name,'h11|h13']] #rename
    #         Group_1_match_df = pd.concat([Group_1_match_df,insert_df1],axis=0) #疊加
    #         i+=1
    #     Group_1_match_df
        
    #     #Hb(g/dL)
    #     Group_2 = ALL_LAB.query(" h18 == \"08003C\" | h18 == \"08011C\" | h18 == \"08012C\" | h18 == \"08082C\" | h18 == \"08014C\"") #query order code
    #     Group_2['r2'] = Group_2['r2'] + "||" + " (" + Group_2['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_2 = Group_2.set_index(keys='id',drop=False)         #選定index
    #     Group_2 = pd.get_dummies(Group_2.r2)                      #one hot
    #     Group_2_list = list(Group_2)                              #取出r2 onehot包含哪些項
    #     Group_2_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_2_list)):
    #         insert_df2 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_2_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_2_list[i].replace('||','')
    #         insert_df2.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df2 = insert_df2[['id',column_name,'h11|h13']] #drop
    #         Group_2_match_df = pd.concat([Group_2_match_df,insert_df2],axis=0) #疊加
    #         i+=1  
    #     Group_2_match_df

    #     #Platelet(103/uL)
    #     Group_3 = ALL_LAB.query(" h18 == \"08006C\" | h18 == \"08011C\"") #query order code
    #     Group_3['r2'] = Group_3['r2'] + "||" + " (" + Group_3['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_3 = Group_3.set_index(keys='id',drop=False)         #選定index
    #     Group_3 = pd.get_dummies(Group_3.r2)                      #one hot
    #     Group_3_list = list(Group_3)                              #取出r2 onehot包含哪些項
    #     Group_3_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_3_list)):
    #         insert_df3 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_3_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_3_list[i].replace('||','')
    #         insert_df3.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df3 = insert_df3[['id',column_name,'h11|h13']] #drop
    #         Group_3_match_df = pd.concat([Group_3_match_df,insert_df3],axis=0) #疊加
    #         i+=1
    #     Group_3_match_df

    #     #AFP(ng/ml)
    #     Group_4 = ALL_LAB.query(" h18 == \"12007C\" | h18 == \"27049C\"") #query order code
    #     Group_4['r2'] = Group_4['r2'] + "||" + " (" + Group_4['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_4 = Group_4.set_index(keys='id',drop=False)         #選定index
    #     Group_4 = pd.get_dummies(Group_4.r2)                      #one hot
    #     Group_4_list = list(Group_4)                              #取出r2 onehot包含哪些項
    #     Group_4_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_4_list)):
    #         insert_df4 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_4_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_4_list[i].replace('||','')
    #         insert_df4.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df4 = insert_df4[['id',column_name,'h11|h13']] #drop
    #         Group_4_match_df = pd.concat([Group_4_match_df,insert_df4],axis=0) #疊加
    #         i+=1
    #     Group_4_match_df

    #     #GOT(U/L)
    #     Group_5 = ALL_LAB.query(" h18 == \"09025C\"") #query order code
    #     Group_5['r2'] = Group_5['r2'] + "||" + " (" + Group_5['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_5 = Group_5.set_index(keys='id',drop=False)         #選定index
    #     Group_5 = pd.get_dummies(Group_5.r2)                      #one hot
    #     Group_5_list = list(Group_5)                              #取出r2 onehot包含哪些項
    #     Group_5_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_5_list)):
    #         insert_df5 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_5_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_5_list[i].replace('||','')
    #         insert_df5.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df5 = insert_df5[['id',column_name,'h11|h13']] #drop
    #         Group_5_match_df = pd.concat([Group_5_match_df,insert_df5],axis=0) #疊加
    #         i+=1
    #     Group_5_match_df

    #     #GPT(U/L)
    #     Group_6 = ALL_LAB.query(" h18 == \"09026C\"") #query order code
    #     Group_6['r2'] = Group_6['r2'] + "||" + " (" + Group_6['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_6 = Group_6.set_index(keys='id',drop=False)         #選定index
    #     Group_6 = pd.get_dummies(Group_6.r2)                      #one hot
    #     Group_6_list = list(Group_6)                              #取出r2 onehot包含哪些項
    #     Group_6_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_6_list)):
    #         insert_df6 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_6_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_6_list[i].replace('||','')
    #         insert_df6.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df6 = insert_df6[['id',column_name,'h11|h13']] #drop
    #         Group_6_match_df = pd.concat([Group_6_match_df,insert_df6],axis=0) #疊加
    #         i+=1
    #     Group_6_match_df
        
    #     #CA19-9(U/mL)
    #     Group_7 = ALL_LAB.query(" h18 == \"12079C\" | h18 == \"27055C\"") #query order code
    #     Group_7['r2'] = Group_7['r2'] + "||" + " (" + Group_7['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_7 = Group_7.set_index(keys='id',drop=False)         #選定index
    #     Group_7 = pd.get_dummies(Group_7.r2)                      #one hot
    #     Group_7_list = list(Group_7)                              #取出r2 onehot包含哪些項
    #     Group_7_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_7_list)):
    #         insert_df7 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_7_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_7_list[i].replace('||','')
    #         insert_df7.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df7 = insert_df7[['id',column_name,'h11|h13']] #drop
    #         Group_7_match_df = pd.concat([Group_7_match_df,insert_df7],axis=0) #疊加
    #         i+=1
    #     Group_7_match_df
        
    #     #CHO(mg/dL)
    #     Group_8 = ALL_LAB.query(" h18 == \"09001C\"") #query order code
    #     Group_8['r2'] = Group_8['r2'] + "||" + " (" + Group_8['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_8 = Group_8.set_index(keys='id',drop=False)         #選定index
    #     Group_8 = pd.get_dummies(Group_8.r2)                      #one hot
    #     Group_8_list = list(Group_8)                              #取出r2 onehot包含哪些項
    #     Group_8_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_8_list)):
    #         insert_df8 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_8_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_8_list[i].replace('||','')
    #         insert_df8.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df8 = insert_df8[['id',column_name,'h11|h13']] #drop
    #         Group_8_match_df = pd.concat([Group_8_match_df,insert_df8],axis=0) #疊加
    #         i+=1
    #     Group_8_match_df
        
    #     #TG(mg/dL)
    #     Group_9 = ALL_LAB.query(" h18 == \"09004C\"") #query order code
    #     Group_9['r2'] = Group_9['r2'] + "||" + " (" + Group_9['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_9 = Group_9.set_index(keys='id',drop=False)         #選定index
    #     Group_9 = pd.get_dummies(Group_9.r2)                      #one hot
    #     Group_9_list = list(Group_9)                              #取出r2 onehot包含哪些項
    #     Group_9_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_9_list)):
    #         insert_df9 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_9_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_9_list[i].replace('||','')
    #         insert_df9.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df9 = insert_df9[['id',column_name,'h11|h13']] #drop
    #         Group_9_match_df = pd.concat([Group_9_match_df,insert_df9],axis=0) #疊加
    #         i+=1
    #     Group_9_match_df

    #     #Albumin(g/dL)
    #     Group_10 = ALL_LAB.query(" h18 == \"09038C\"") #query order code
    #     Group_10['r2'] = Group_10['r2'] + "||" + " (" + Group_10['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_10 = Group_10.set_index(keys='id',drop=False)         #選定index
    #     Group_10 = pd.get_dummies(Group_10.r2)                      #one hot
    #     Group_10_list = list(Group_10)                              #取出r2 onehot包含哪些項
    #     Group_10_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_10_list)):
    #         insert_df10 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_10_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_10_list[i].replace('||','')
    #         insert_df10.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df10 = insert_df10[['id',column_name,'h11|h13']] #drop
    #         Group_10_match_df = pd.concat([Group_10_match_df,insert_df10],axis=0) #疊加
    #         i+=1
    #     Group_10_match_df

    #     #BUN(mg/dL)
    #     Group_11 = ALL_LAB.query(" h18 == \"09002C\"") #query order code
    #     Group_11['r2'] = Group_11['r2'] + "||" + " (" + Group_11['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_11 = Group_11.set_index(keys='id',drop=False)         #選定index
    #     Group_11 = pd.get_dummies(Group_11.r2)                      #one hot
    #     Group_11_list = list(Group_11)                              #取出r2 onehot包含哪些項
    #     Group_11_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_11_list)):
    #         insert_df11 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_11_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_11_list[i].replace('||','')
    #         insert_df11.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df11 = insert_df11[['id',column_name,'h11|h13']] #drop
    #         Group_11_match_df = pd.concat([Group_11_match_df,insert_df11],axis=0) #疊加
    #         i+=1
    #     Group_11_match_df

    #     #GFR(ml/min/1.73m^)
    #     Group_12 = ALL_LAB.query(" h18 == \"09015C\" | h18 == \"09016C\"") #query order code
    #     Group_12['r2'] = Group_12['r2'] + "||" + " (" + Group_12['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_12 = Group_12.set_index(keys='id',drop=False)         #選定index
    #     Group_12 = pd.get_dummies(Group_12.r2)                      #one hot
    #     Group_12_list = list(Group_12)                              #取出r2 onehot包含哪些項
    #     Group_12_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_12_list)):
    #         insert_df12 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_12_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_12_list[i].replace('||','')
    #         insert_df12.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df12 = insert_df12[['id',column_name,'h11|h13']] #drop
    #         Group_12_match_df = pd.concat([Group_12_match_df,insert_df12],axis=0) #疊加
    #         i+=1
    #     Group_12_match_df

    #     #Creatinine(urine)(mg/dL)
    #     Group_13 = ALL_LAB.query(" h18 == \"09016C\"") #query order code
    #     Group_13['r2'] = Group_13['r2'] + "||" + " (" + Group_13['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_13 = Group_13.set_index(keys='id',drop=False)         #選定index
    #     Group_13 = pd.get_dummies(Group_13.r2)                      #one hot
    #     Group_13_list = list(Group_13)                              #取出r2 onehot包含哪些項
    #     Group_13_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_13_list)):
    #         insert_df13 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_13_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_13_list[i].replace('||','')
    #         insert_df13.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df13 = insert_df13[['id',column_name,'h11|h13']] #drop
    #         Group_13_match_df = pd.concat([Group_13_match_df,insert_df13],axis=0) #疊加
    #         i+=1
    #     Group_13_match_df

    #     #PSA(ng/mL)
    #     Group_14 = ALL_LAB.query(" h18 == \"12081C\" | h18 == \"27052C\"") #query order code
    #     Group_14['r2'] = Group_14['r2'] + "||" + " (" + Group_14['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_14 = Group_14.set_index(keys='id',drop=False)         #選定index
    #     Group_14 = pd.get_dummies(Group_14.r2)                      #one hot
    #     Group_14_list = list(Group_14)                              #取出r2 onehot包含哪些項
    #     Group_14_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_14_list)):
    #         insert_df14 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_14_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_14_list[i].replace('||','')
    #         insert_df14.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df14 = insert_df14[['id',column_name,'h11|h13']] #drop
    #         Group_14_match_df = pd.concat([Group_14_match_df,insert_df14],axis=0) #疊加
    #         i+=1
    #     Group_14_match_df     

    #     dfs=[]
    #     if Group_1_match_df.empty == False :
    #         dfs.append(Group_1_match_df)
    #     if Group_2_match_df.empty == False :
    #         dfs.append(Group_2_match_df)
    #     if Group_3_match_df.empty == False :
    #         dfs.append(Group_3_match_df)
    #     if Group_4_match_df.empty == False :
    #         dfs.append(Group_4_match_df)
    #     if Group_5_match_df.empty == False :
    #         dfs.append(Group_5_match_df)
    #     if Group_6_match_df.empty == False :
    #         dfs.append(Group_6_match_df)
    #     if Group_7_match_df.empty == False :
    #         dfs.append(Group_7_match_df)
    #     if Group_8_match_df.empty == False :
    #         dfs.append(Group_8_match_df)
    #     if Group_9_match_df.empty == False :
    #         dfs.append(Group_9_match_df)
    #     if Group_10_match_df.empty == False :
    #         dfs.append(Group_10_match_df)
    #     if Group_11_match_df.empty == False :
    #         dfs.append(Group_11_match_df)
    #     if Group_12_match_df.empty == False :
    #         dfs.append(Group_12_match_df)
    #     if Group_13_match_df.empty == False :
    #         dfs.append(Group_13_match_df)
    #     if Group_14_match_df.empty == False :
    #         dfs.append(Group_14_match_df)          

    #     try:
    #         del [[Group_1_match_df, Group_2_match_df, Group_3_match_df, Group_4_match_df, Group_5_match_df, Group_6_match_df, Group_7_match_df, 
    #         Group_8_match_df, Group_9_match_df, Group_10_match_df, Group_11_match_df, Group_12_match_df, Group_13_match_df, Group_14_match_df]]
    #         gc.collect()
    #     except:
    #         pass

    #     df_final = ft.reduce(lambda left, right: pd.merge(left, right, on=['id','h11|h13'],how='outer',suffixes=('', '_delme')), dfs)
    #     df_final = df_final[[d for d in df_final.columns if not d.endswith('_delme')]]
    #     df_final = pd.merge(df_final, CRLF, on=['id'],how='left')
    #     df_final = df_final.drop_duplicates()
    #     # df_final = pd.merge(df_final, AE_row_count_before, on=['id'],how='left')
    #     # df_final = df_final.drop_duplicates()
    #     # df_final = pd.merge(df_final, BE_row_count_before, on=['id'],how='left',suffixes=('_AE', '_BE'))
    #     # df_final = df_final.drop_duplicates()
    #     df_final
        
    # except:
    #     df_final = pd.DataFrame({'no_match':[' no matching results']})
    df_final = pd.DataFrame({'stop':[' stop used']})
    ####################### recoding ##############################

    try:
        df_CRLF['age_group'] = df_CRLF['age'].apply(age_group)
    except:
        pass
    
    ###80%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('80%')
    f.close()
    ###80%進度點###

    ####################### demographic ##############################

    try:
        df_CRLF_demo = describe(df_CRLF).T
        df_CRLF_demo = missing(df_CRLF_demo, df_CRLF)
        df_CRLF_demo_c = split_v_c(df_CRLF_demo)[1] #[0]是跑連續數值
    except:
        df_CRLF_demo_c = pd.DataFrame({'no_match':['The data cannot be calculated demographic']})

    ###90%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('90%')
    f.close()
    ###90%進度點###

    ####################### Death ##############################

    df_DEATH = pd.read_sql("SELECT id,d2,d3,d4,d5,d6,d7 FROM " + "[" + "DEATH" + "]", conn)
    df_DEATH = pd.merge(df_CRLF_cohort, df_DEATH, how='inner', on=['id'], indicator=False)
    df_DEATH.drop('didiag',axis=1,inplace=True)
    if df_DEATH.empty == True:
        df_DEATH = pd.DataFrame({'no_match':['not found cohort']})

    conn.close()

    return(df_CRLF, AE_row_count_before, AE_row_count_after, BE_row_count_before, BE_row_count_after, 
           TOTFAO1_match_df, TOTFBO1_matchD_df, TOTFBO1_matchS_df,TOTFBO1_matchC_df, LABM1_Check_match_df,
           LABM1_Surgery_match_df, df_CASE_match_df,df_final,df_CRLF_demo_c,df_DEATH)

def C_CANCER_plus(logic_structure):

    #DB
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor
    #定位
    sort = logic_structure.get("table_select")
    i=0
    global table_list
    table_list = []
    for i in range(len(sort)):
        table_list.append(sort[i]['table'])
        i=i+1   
     #common function
    def cut_date(x):
        try:          
            if len(x)>8:
                x = x[0:8]
            x_f = x[0:4]
            x_f = x_f.replace('9999','1900')
            x_m = x[4:6]
            x_m = x_m.replace('99','01')
            x_b = x[6:8]
            x_b = x_b.replace('99','01')
            x = x_f + x_m + x_b
            x = parse(x)
            return x
        
        except:
            x_error = '19000101'
            x_error = parse(x_error)
            return x_error

    #cohrot 重要項 
    disease = logic_structure['disease']
    keep = logic_structure['keep']
    search_id = logic_structure['search_id']
    pattern = r'"([^"]+)"'
    search_id2 = re.findall(pattern, search_id)

    #write index path
    index_write = str(logic_structure['index_write'])
    is_coop = str(logic_structure['coop'])
    if is_coop == "0":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer\\" + index_write +"\\config\\"
    if is_coop == "1":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop\\" + index_write +"\\config\\"
    ###0%進度點###
    process_path = plug_path+'C_process.txt'
    try:
        os.remove(process_path)
    except:
        pass
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('0%')
    f.close()
    ###0%進度點###

    # CRLF ############################篩選結果資料(大excel用)、帶時間ID冊(日期用、可能與各表交集用)
    #astype and site query
    CRLF_index = table_list.index('CRLF')
    # df_CRLF = pd.read_sql("SELECT id,sex,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM " + "CRLF", conn)
    df_CRLF = pd.read_sql("SELECT * FROM " + "CRLF", conn)
    df_CRLF = df_CRLF.astype(str)
    df_CRLF['age'] = df_CRLF['age'].astype(int)
    df_CRLF = df_CRLF.query(search_id,engine='python')
    df_CRLF = df_CRLF.query(disease,engine='python')
    # if df_CRLF.empty == True:
    #     ###離開100%進度點###
    #     process_path = plug_path+'C_process.txt'
    #     f = open(process_path, 'w')
    #     f.write('100%')
    #     f.close()
    #     ###離開100%進度點###
    #     nomatch_df = pd.DataFrame({'no_match':['CRLF Site code no matching results']})
    #     return(nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df)
    
    #queryList(複數個欄位):篩選結果資料
    i=0
    for each_query in range(len(logic_structure['table_select'][CRLF_index]['queryList'])):
        each = list(logic_structure['table_select'][CRLF_index]['queryList'].values())[i][0]
        df_CRLF = df_CRLF.query(each)
        i+=1
    
    # if df_CRLF.empty == True:
    #     nomatch_df = pd.DataFrame({'no_match':['CRLF Columns condition code no matching results']})
    #     ###離開100%進度點###
    #     process_path = plug_path+'C_process.txt'
    #     f = open(process_path, 'w')
    #     f.write('100%')
    #     f.close()
    #     ###離開100%進度點###
    #     return(nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df,nomatch_df)
    
    df_CRLF #CRLF結果
    ##0503更新補癌登沒有的人
    currentDateTime = datetime.datetime.now()
    date = currentDateTime.date()
    year = date.strftime("%Y"+"0630")

    list_origquery = df_CRLF['id'].to_list()
    list_intersection = set(search_id2) - set(list_origquery)
    list_intersection = list(list_intersection)
    df_list_intersection = pd.DataFrame(list_intersection)
    df_list_intersection.rename(columns = {0:'id'}, inplace = True)
    year = str(int(year)-10000) #20230630-10000
    df_list_intersection['didiag'] = year
    
    df_CRLF = pd.concat([df_CRLF,df_list_intersection], axis=0, ignore_index=True)

    #cohort_time:帶時間ID冊
    df_CRLF_cohort = df_CRLF[['id','didiag']]
    df_CRLF_cohort['didiag'] = df_CRLF_cohort['didiag'].apply(cut_date)
    df_CRLF_cohort

    # #存活ID冊
    # df_CRLF_ID = df_CRLF_cohort['id'].drop_duplicates().tolist()
    # df_CRLF_ID
 
    ###10%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('10%')
    f.close()
    ###10%進度點###

 ######################## TOTFAE ########################## 正常篩選結果、無效回報

    #function
    def TOTFAE_count(Col_Name, ICD):
            _19 = df_TOTFAE.loc[df_TOTFAE['d19'].str.startswith((ICD), na = False)] #各欄找ICD
            _20 = df_TOTFAE.loc[df_TOTFAE['d20'].str.startswith((ICD), na = False)]
            _21 = df_TOTFAE.loc[df_TOTFAE['d21'].str.startswith((ICD), na = False)]
            _22 = df_TOTFAE.loc[df_TOTFAE['d22'].str.startswith((ICD), na = False)]
            _23 = df_TOTFAE.loc[df_TOTFAE['d23'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_19,_20,_21,_22,_23])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            AE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            AE_row_count = AE_row_count.to_frame()
            AE_row_count = AE_row_count.reset_index()
            AE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return AE_row_count
        
    #comobidity
    TOTFAE_index = table_list.index('TOTFAE')
    comobidity_num = len(logic_structure['table_select'][TOTFAE_index]['logic_before'])
    if comobidity_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFAE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        AE_row_count_before = pd.DataFrame({'id':['N/A'],'comobidity_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        # df_TOTFAE = pd.read_sql("SELECT * FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_CRLF_cohort, df_TOTFAE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFAE.empty == True:
            AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE內 無同癌登的人群']})
        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['didiag'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600
                
            #before
            df_TOTFAE_before = df_TOTFAE.query("diff_days <=" + str(start_time_TOTFAE) + " " + "and" + " " + "diff_days >=" + str(0))
            
            if df_TOTFAE_before.empty == True:
                AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity time no matching results']})
    
            else:
                comobidity_col_name = []    #收集config中的所有comobidity_name
                comobidity_condition =[]    #收集config中的所有comobidity_condition

                for c in range(comobidity_num):
                    comobidity_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['col_name'])
                    comobidity_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['condition'])

                c_ = 0
                for c_ in range(comobidity_num):
                    if c_ == 0:
                        AE_row_count_before = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                    else:
                        AE_row_count_before_append = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                        AE_row_count_before = pd.merge(AE_row_count_before, AE_row_count_before_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d=0
                # for d in range(len(comobidity_col_name)):
                #     AE_row_count_before = AE_row_count_before.drop(AE_row_count_before[AE_row_count_before[comobidity_col_name[d]]<2].index) #刪除門診<2的人

                if AE_row_count_before.empty == True:
                    AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity count no matching results with >=2']})
                    
                else:
                    TOTFAE_ID_before = AE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    AE_row_count_before #輸出結果表
                    
    #complication
    TOTFAE_index = table_list.index('TOTFAE')
    complication_num = len(logic_structure['table_select'][TOTFAE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFAE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        AE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_CRLF_cohort, df_TOTFAE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFAE.empty == True:
            AE_row_count_after  = pd.DataFrame({'no_match':['TOTFAE內 無同癌登的人群']})
            
        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['didiag'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600
                
            #after
            df_TOTFAE_after = df_TOTFAE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAE))
            
            if df_TOTFAE_after.empty == True:
                AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication time no matching results']})
    
            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition
                
                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['condition'])
       
                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        AE_row_count_after = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        AE_row_count_after_append = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        AE_row_count_after = pd.merge(AE_row_count_after, AE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d = 0
                # for d in range(len(complication_col_name)):
                #     AE_row_count_after = AE_row_count_after.drop(AE_row_count_after[AE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人
                    
                if AE_row_count_after.empty == True:
                    AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication count no matching results with >=2']})
                    
                else:
                    TOTFAE_ID_after = AE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    AE_row_count_after #輸出結果表

    ###20%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('20%')
    f.close()
    ###20%進度點###

 ######################## TOTFBE ########################## 正常篩選結果、無效回報
    #function
    def TOTFBE_count(Col_Name, ICD):
            _25 = df_TOTFBE.loc[df_TOTFBE['d25'].str.startswith((ICD), na = False)] #各欄找ICD
            _26 = df_TOTFBE.loc[df_TOTFBE['d26'].str.startswith((ICD), na = False)]
            _27 = df_TOTFBE.loc[df_TOTFBE['d27'].str.startswith((ICD), na = False)]
            _28 = df_TOTFBE.loc[df_TOTFBE['d28'].str.startswith((ICD), na = False)]
            _29 = df_TOTFBE.loc[df_TOTFBE['d29'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_25,_26,_27,_28,_29])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            BE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            BE_row_count = BE_row_count.to_frame()
            BE_row_count = BE_row_count.reset_index()
            BE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return BE_row_count
        
    #comobidity
    TOTFBE_index = table_list.index('TOTFBE')
    comobidity_num = len(logic_structure['table_select'][TOTFBE_index]['logic_before'])
    
    if comobidity_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        BE_row_count_before = pd.DataFrame({'id':['N/A'],'comorbidity_N/A':['N/A']}) 
        
    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_CRLF_cohort, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date)
        
        if df_TOTFBE.empty == True:
            BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
            BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
        else:
            
            df_TOTFBE['diff_days'] = df_TOTFBE['didiag'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600
            
            #before
            df_TOTFBE_before = df_TOTFBE.query("diff_days <=" + str(start_time_TOTFBE) + " " + "and" + " " + "diff_days >=" + str(0))
            
            if df_TOTFBE_before.empty == True:
                BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity time no matching results']})
            else:
                comobidity_col_name = []    #收集config中的所有comobidity_name
                comobidity_condition =[]    #收集config中的所有comobidity_condition
                
                for c in range(comobidity_num):
                    comobidity_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['col_name'])
                    comobidity_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['condition'])
                c_ = 0
                for c_ in range(comobidity_num):
                    if c_ == 0:
                        BE_row_count_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_append_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                        BE_row_count_before = pd.merge(BE_row_count_before, BE_row_count_append_before, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1

                # d = 0
                # for d in range(len(comobidity_col_name)):
                #     BE_row_count_before = BE_row_count_before.drop(BE_row_count_before[BE_row_count_before[comobidity_col_name[d]]<1].index) #刪除住院<1的人

                if BE_row_count_before.empty == True:
                    BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity count no matching results with >=1']})
                    
                else:
                    TOTFBE_ID_before = BE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_before #輸出結果表
                    
    #complication
    TOTFBE_index = table_list.index('TOTFBE')
    complication_num = len(logic_structure['table_select'][TOTFBE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_CRLF[['id']] #存活ID冊(沒被篩)
        BE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_CRLF_cohort, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFBE.empty == True:
            BE_row_count_after  = pd.DataFrame({'no_match':['TOTFBE內 無同癌登的人群']})
            
        else:
            df_TOTFBE['diff_days'] = df_TOTFBE['didiag'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')
            
            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600
                
            #after
            df_TOTFBE_after = df_TOTFBE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBE))
            
            if df_TOTFBE_after.empty == True:
                BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication time no matching results']})
    
            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition

                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['condition'])
                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        BE_row_count_after = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_after_append = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        BE_row_count_after = pd.merge(BE_row_count_after, BE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1
                
                # d = 0
                # for d in range(len(complication_col_name)):
                #     BE_row_count_after = BE_row_count_after.drop(BE_row_count_after[BE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人
                    
                if BE_row_count_after.empty == True:
                    BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication count no matching results with >=2']})
                    
                else:
                    TOTFBE_ID_after = BE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_after #輸出結果表

    ###30%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('30%')
    f.close()
    ###30%進度點###

    # TOTFA01 ######################### 篩選結果存活ID冊、藥品細項清單                
    TOTFAO1_index = table_list.index('TOTFAO1')
     # df_TOTFAO1 = pd.read_sql("SELECT d3,p4,d9 FROM " + "TOTFAO1", conn)
    df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "TOTFAO1", conn)
    df_TOTFAO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFAO1 = pd.merge(df_CRLF_cohort, df_TOTFAO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFAO1['d9'] = df_TOTFAO1['d9'].apply(cut_date)

    if df_TOTFAO1.empty == True:
        TOTFAO1_ID = df_CRLF[['id']]
        TOTFAO1_match_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p4':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p9':['N/A'],'p10':['N/A'],'p13':['N/A'],'p14':['N/A'],'p15':['N/A'],'p17':['N/A'],'d9':['N/A']})
    else:
        df_TOTFAO1['diff_days'] = df_TOTFAO1['didiag'] - df_TOTFAO1['d9']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFAO1['diff_days'] = df_TOTFAO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFAO1 = int(logic_structure['table_select'][TOTFAO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFAO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFAO1 = -(int(logic_structure['table_select'][TOTFAO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFAO1 = -3600

        df_TOTFAO1 = df_TOTFAO1.query("diff_days <=" + str(start_time_TOTFAO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAO1))#動時間還未進藥篩
        
        if df_TOTFAO1.empty == True:
            TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug time no matching results']})
            
        else:
            TOTFAO1_match_df = pd.DataFrame()
            query_TOTFAO1 = logic_structure['table_select'][TOTFAO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFAO1_batch = query_TOTFAO1.split('|')

            try: #防空表
                i=0
                for A in range(len(query_TOTFAO1_batch)):
                    insert_df = df_TOTFAO1.query(query_TOTFAO1_batch[i])
                    TOTFAO1_match_df = pd.concat([TOTFAO1_match_df, insert_df], axis=0)
                    TOTFAO1_match_df = TOTFAO1_match_df.drop_duplicates()
                    i+=1

                if TOTFAO1_match_df.empty == True:
                    TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})
                    
                else:
                    #藥品細項清單
                    TOTFAO1_match_df

                    #存活ID冊
                    TOTFAO1_ID = TOTFAO1_match_df['id'].drop_duplicates().tolist()
            except:
                TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})
                TOTFAO1_ID=[]

    ###40%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('40%')
    f.close()
    ###40%進度點###

    # TOTFB01 ######################### 篩選結果存活ID冊、醫令細項清單
    df_TOTFBO1_sql = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)            
    #藥物
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchD_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchD_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchD_df = pd.concat([TOTFBO1_matchD_df, insert_df], axis=0)
                    TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchD_df.empty == True:
                    TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     
                else:
                    #醫令細項清單
                    TOTFBO1_matchD_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_D = TOTFBO1_matchD_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_D
            except:
                TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_D=[]
                
    #手術
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchS_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_surgery'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_surgery'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchS_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_surgery']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchS_df = pd.concat([TOTFBO1_matchS_df, insert_df], axis=0)
                    TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchS_df.empty == True:
                    TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     

                else:
                    #醫令細項清單
                    TOTFBO1_matchS_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_S = TOTFBO1_matchS_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_S
            except:
                TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_S=[]

    #檢驗檢查
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    # df_TOTFBO1 = pd.read_sql("SELECT d3,p3,d10 FROM " + "TOTFBO1", conn)
    # df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_CRLF_cohort, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_CRLF[['id']]
        TOTFBO1_matchC_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 
        
    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['didiag'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')
        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_check'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_check'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩
        
        if df_TOTFBO1.empty == True:
            TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})
            
        else:     
            TOTFBO1_matchC_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_check']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')

            try: 
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchC_df = pd.concat([TOTFBO1_matchC_df, insert_df], axis=0)
                    TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop_duplicates()
                    i+=1
                
                if TOTFBO1_matchC_df.empty == True:
                    TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
     

                else:
                    #醫令細項清單
                    TOTFBO1_matchC_df    
                    
                    #存活ID冊
                    TOTFBO1_ID_C = TOTFBO1_matchC_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_C
            except:
                TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_C=[]

    ###50%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('50%')
    f.close()
    ###50%進度點###

    # LABM1_Check ########################### 篩選結果存活ID冊、檢驗細項清單
    LABM1_Check_index = table_list.index('LABM1_Check')
    df_LABM1_Check = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11'] + df_LABM1_Check['h13']
    df_LABM1_Check.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Check = pd.merge(df_CRLF_cohort, df_LABM1_Check, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11|h13'].apply(cut_date)

    if df_LABM1_Check.empty == True:
        LABM1_Check_ID = df_CRLF[['id']]
        LABM1_Check_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'Index':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    
    else:
        df_LABM1_Check['diff_days'] = df_LABM1_Check['didiag'] - df_LABM1_Check['h11|h13']  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Check['diff_days'] = df_LABM1_Check['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Check = int(logic_structure['table_select'][LABM1_Check_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Check = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Check = -(int(logic_structure['table_select'][LABM1_Check_index]['end_time'])*30)
        except:
            end_time_LABM1_Check = -3600

        df_LABM1_Check = df_LABM1_Check.query("diff_days <=" + str(start_time_LABM1_Check) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Check))
        
        if df_LABM1_Check.empty == True:
            LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check time no matching results']})
        
        else:
            LABM1_Check_match_df = pd.DataFrame()
            query_LABM1_Check = logic_structure['table_select'][LABM1_Check_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Check_batch = query_LABM1_Check.split('|')

            try:

                i=0
                for LC in range(len(query_LABM1_Check_batch)):
                    insert_df = df_LABM1_Check.query(query_LABM1_Check_batch[i])
                    LABM1_Check_match_df = pd.concat([LABM1_Check_match_df, insert_df], axis=0)
                    LABM1_Check_match_df = LABM1_Check_match_df.drop_duplicates()
                    i+=1
                    
                if LABM1_Check_match_df.empty == True:
                    LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})

                else:
                    #檢驗細項清單
                    LABM1_Check_match_df
        
                    #存活ID冊
                    LABM1_Check_ID = LABM1_Check_match_df['id'].drop_duplicates().tolist()
                    LABM1_Check_ID
            except:
                LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})
                LABM1_Check_ID=[]
    # LABM1_Surgery ########################### 篩選結果存活ID冊、手術細項清單
    LABM1_Surgery_index = table_list.index('LABM1_Surgery')
    df_LABM1_Surgery = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11'] + df_LABM1_Surgery['h13']
    df_LABM1_Surgery.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Surgery = pd.merge(df_CRLF_cohort, df_LABM1_Surgery, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11|h13'].apply(cut_date)

    if df_LABM1_Surgery.empty == True :
        LABM1_Surgery_ID = df_CRLF[['id']]
        LABM1_Surgery_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'Index':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    else:
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['didiag'] - df_LABM1_Surgery['h11|h13']                                  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Surgery = int(logic_structure['table_select'][LABM1_Surgery_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Surgery = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Surgery = -(int(logic_structure['table_select'][LABM1_Surgery_index]['end_time'])*30)
        except:
            end_time_LABM1_Surgery = -3600

        df_LABM1_Surgery = df_LABM1_Surgery.query("diff_days <=" + str(start_time_LABM1_Surgery) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Surgery))
        
        if df_LABM1_Surgery.empty == True:
            LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery time no matching results']})
        
        else:
            
            LABM1_Surgery_match_df = pd.DataFrame()
            query_LABM1_Surgery = logic_structure['table_select'][LABM1_Surgery_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Surgery_batch = query_LABM1_Surgery.split('|')

            try:

                i=0
                for LS in range(len(query_LABM1_Surgery_batch)):
                    insert_df = df_LABM1_Surgery.query(query_LABM1_Surgery_batch[i])
                    LABM1_Surgery_match_df = pd.concat([LABM1_Surgery_match_df, insert_df], axis=0)
                    LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop_duplicates()
                    i+=1
                
                if LABM1_Surgery_match_df.empty == True:
                    LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})
     
                else:
                    
                    #手術細項清單
                    LABM1_Surgery_match_df

                    #存活ID冊
                    LABM1_Surgery_ID = LABM1_Surgery_match_df['id'].drop_duplicates().tolist()
                    LABM1_Surgery_ID
            except:
                LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})
                LABM1_Surgery_ID=[]

    ###60%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('60%')
    f.close()
    ###60%進度點###

    # CASE ############################ 篩選結果存活ID冊、結果檔            
    CASE_index = table_list.index('CASE')
    df_CASE = pd.read_sql("SELECT * FROM " + "[" + "CASE" + "]", conn)
    df_CASE = pd.merge(df_CRLF_cohort, df_CASE, how='left', on=['id'], indicator=False)
    
    if df_CASE.empty == True :
        df_CASE_ID = df_CRLF[['id']]
        df_CASE_match_df = pd.DataFrame({'Index':['N/A'],'id':['N/A'],'gender':['N/A'],'m2':['N/A'],'m3':['N/A'],'m4':['N/A'],'m5':['N/A'],'m6':['N/A'],'m7':['N/A'],'verify':['N/A'],'IsUploadHash':['N/A'],'CreateTime':['N/A'],'ModifyTime':['N/A'],'h11|h13':['N/A']}) 
    
    else:
    
        i=0
        for each_query in range(len(logic_structure['table_select'][CASE_index]['queryList'])):
            each = list(logic_structure['table_select'][CASE_index]['queryList'].values())[i][0]
            df_CASE_match_df = df_CASE.query(each)
            i+=1
        
        if df_CASE_match_df.empty == True:
            df_CASE_match_df = pd.DataFrame({'no_match':['CASE Columns condition code no matching results']})
        
        else:
            #CASE結果
            df_CASE_match_df = df_CASE_match_df.drop(['d3'],axis=1)
            df_CASE_match_df

            #存活ID冊
            df_CASE_ID = df_CASE_match_df['id'].drop_duplicates().tolist()
            df_CASE_ID

    # normal-output ############################        
    try:
        df_CRLF = df_CRLF[keep]
    except:
        df_CRLF = df_CRLF
        
    try:
        AE_row_count_before = AE_row_count_before
    except:
        AE_row_count_before = AE_row_count_before
    
    try:
        AE_row_count_after = AE_row_count_after     
    except:
        AE_row_count_after = AE_row_count_after
    
    try:
        BE_row_count_before = BE_row_count_before   
    except:
        BE_row_count_before = BE_row_count_before

    try:
        BE_row_count_after = BE_row_count_after   
    except:
        BE_row_count_after = BE_row_count_after

    try:
        TOTFAO1_match_df = TOTFAO1_match_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFAO1_match_df = TOTFAO1_match_df
    
    try:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df
    
    try:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df
    
    try:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df
                
    try:
        LABM1_Check_match_df = LABM1_Check_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Check_match_df = LABM1_Check_match_df
        
    try:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df
        
    try:
        df_CASE_match_df = df_CASE_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1)
    except:
        df_CASE_match_df = df_CASE_match_df
    
    ###70%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('70%')
    f.close()
    ###70%進度點###
    
 # HUANG ###############################黃醫師
    # try:
    #     def r2_to_command(x):                            
    #         x = "r2 == " + '\"' + x + '\"'
    #         return(x)

    #     # ALL_LAB = pd.read_sql("SELECT h9,h18,r2,r4,r5,h11,h13 FROM " + "LABM1", conn)
    #     # ALL_LAB['h11|h13'] = ALL_LAB['h11'] + ALL_LAB['h13']
    #     # CRLF = pd.read_sql("SELECT id,gender,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM"  + " CRLF", conn)
    #     ALL_LAB = LABM1_Check_match_df
    #     CRLF = pd.read_sql("SELECT id,sex,age,hist,behavior,smoking,btchew,drinking,ct,cn,cm,cstage,pt,pn,pm,pstage,size,grade_c,grade_p,lateral,site,lvi,pni,smargin,smargin_d,srs,sls,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM " + "CRLF", conn)
    #     # CRLF = pd.read_sql("SELECT id,gender,age,site,didiag,grade_c,grade_p,ct,cn,cm,cstage,pt,pn,pm,ps,pstage,size,hist,behavior,lvi,pni,smargin,smargin_d,srs,sls,smoking,drinking,ssf1,ssf2,ssf3,ssf4,ssf5,ssf6,ssf7,ssf8,ssf9,ssf10 FROM"  + " CRLF", conn)
        
    #     #CEA(ng/ml)
    #     Group_1 = ALL_LAB.query(" h18 == \"12021C\" | h18 == \"27050C\"") #query order code
    #     Group_1['r2'] = Group_1['r2'] + "||" + " (" + Group_1['r5'] + ")"
    #     Group_1 = Group_1.set_index(keys='id',drop=False)         #選定index
    #     Group_1 = pd.get_dummies(Group_1.r2)                      #one hot
    #     Group_1_list = list(Group_1)                              #取出r2 onehot包含哪些項
    #     Group_1_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_1_list)):
    #         insert_df1 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_1_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_1_list[i].replace('||','')
    #         insert_df1.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df1 = insert_df1[['id',column_name,'h11|h13']] #rename
    #         Group_1_match_df = pd.concat([Group_1_match_df,insert_df1],axis=0) #疊加
    #         i+=1
    #     Group_1_match_df
        
    #     #Hb(g/dL)
    #     Group_2 = ALL_LAB.query(" h18 == \"08003C\" | h18 == \"08011C\" | h18 == \"08012C\" | h18 == \"08082C\" | h18 == \"08014C\"") #query order code
    #     Group_2['r2'] = Group_2['r2'] + "||" + " (" + Group_2['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_2 = Group_2.set_index(keys='id',drop=False)         #選定index
    #     Group_2 = pd.get_dummies(Group_2.r2)                      #one hot
    #     Group_2_list = list(Group_2)                              #取出r2 onehot包含哪些項
    #     Group_2_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_2_list)):
    #         insert_df2 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_2_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_2_list[i].replace('||','')
    #         insert_df2.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df2 = insert_df2[['id',column_name,'h11|h13']] #drop
    #         Group_2_match_df = pd.concat([Group_2_match_df,insert_df2],axis=0) #疊加
    #         i+=1  
    #     Group_2_match_df

    #     #Platelet(103/uL)
    #     Group_3 = ALL_LAB.query(" h18 == \"08006C\" | h18 == \"08011C\"") #query order code
    #     Group_3['r2'] = Group_3['r2'] + "||" + " (" + Group_3['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_3 = Group_3.set_index(keys='id',drop=False)         #選定index
    #     Group_3 = pd.get_dummies(Group_3.r2)                      #one hot
    #     Group_3_list = list(Group_3)                              #取出r2 onehot包含哪些項
    #     Group_3_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_3_list)):
    #         insert_df3 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_3_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_3_list[i].replace('||','')
    #         insert_df3.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df3 = insert_df3[['id',column_name,'h11|h13']] #drop
    #         Group_3_match_df = pd.concat([Group_3_match_df,insert_df3],axis=0) #疊加
    #         i+=1
    #     Group_3_match_df

    #     #AFP(ng/ml)
    #     Group_4 = ALL_LAB.query(" h18 == \"12007C\" | h18 == \"27049C\"") #query order code
    #     Group_4['r2'] = Group_4['r2'] + "||" + " (" + Group_4['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_4 = Group_4.set_index(keys='id',drop=False)         #選定index
    #     Group_4 = pd.get_dummies(Group_4.r2)                      #one hot
    #     Group_4_list = list(Group_4)                              #取出r2 onehot包含哪些項
    #     Group_4_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_4_list)):
    #         insert_df4 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_4_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_4_list[i].replace('||','')
    #         insert_df4.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df4 = insert_df4[['id',column_name,'h11|h13']] #drop
    #         Group_4_match_df = pd.concat([Group_4_match_df,insert_df4],axis=0) #疊加
    #         i+=1
    #     Group_4_match_df

    #     #GOT(U/L)
    #     Group_5 = ALL_LAB.query(" h18 == \"09025C\"") #query order code
    #     Group_5['r2'] = Group_5['r2'] + "||" + " (" + Group_5['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_5 = Group_5.set_index(keys='id',drop=False)         #選定index
    #     Group_5 = pd.get_dummies(Group_5.r2)                      #one hot
    #     Group_5_list = list(Group_5)                              #取出r2 onehot包含哪些項
    #     Group_5_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_5_list)):
    #         insert_df5 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_5_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_5_list[i].replace('||','')
    #         insert_df5.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df5 = insert_df5[['id',column_name,'h11|h13']] #drop
    #         Group_5_match_df = pd.concat([Group_5_match_df,insert_df5],axis=0) #疊加
    #         i+=1
    #     Group_5_match_df

    #     #GPT(U/L)
    #     Group_6 = ALL_LAB.query(" h18 == \"09026C\"") #query order code
    #     Group_6['r2'] = Group_6['r2'] + "||" + " (" + Group_6['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_6 = Group_6.set_index(keys='id',drop=False)         #選定index
    #     Group_6 = pd.get_dummies(Group_6.r2)                      #one hot
    #     Group_6_list = list(Group_6)                              #取出r2 onehot包含哪些項
    #     Group_6_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_6_list)):
    #         insert_df6 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_6_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_6_list[i].replace('||','')
    #         insert_df6.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df6 = insert_df6[['id',column_name,'h11|h13']] #drop
    #         Group_6_match_df = pd.concat([Group_6_match_df,insert_df6],axis=0) #疊加
    #         i+=1
    #     Group_6_match_df
        
    #     #CA19-9(U/mL)
    #     Group_7 = ALL_LAB.query(" h18 == \"12079C\" | h18 == \"27055C\"") #query order code
    #     Group_7['r2'] = Group_7['r2'] + "||" + " (" + Group_7['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_7 = Group_7.set_index(keys='id',drop=False)         #選定index
    #     Group_7 = pd.get_dummies(Group_7.r2)                      #one hot
    #     Group_7_list = list(Group_7)                              #取出r2 onehot包含哪些項
    #     Group_7_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_7_list)):
    #         insert_df7 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_7_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_7_list[i].replace('||','')
    #         insert_df7.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df7 = insert_df7[['id',column_name,'h11|h13']] #drop
    #         Group_7_match_df = pd.concat([Group_7_match_df,insert_df7],axis=0) #疊加
    #         i+=1
    #     Group_7_match_df
        
    #     #CHO(mg/dL)
    #     Group_8 = ALL_LAB.query(" h18 == \"09001C\"") #query order code
    #     Group_8['r2'] = Group_8['r2'] + "||" + " (" + Group_8['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_8 = Group_8.set_index(keys='id',drop=False)         #選定index
    #     Group_8 = pd.get_dummies(Group_8.r2)                      #one hot
    #     Group_8_list = list(Group_8)                              #取出r2 onehot包含哪些項
    #     Group_8_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_8_list)):
    #         insert_df8 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_8_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_8_list[i].replace('||','')
    #         insert_df8.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df8 = insert_df8[['id',column_name,'h11|h13']] #drop
    #         Group_8_match_df = pd.concat([Group_8_match_df,insert_df8],axis=0) #疊加
    #         i+=1
    #     Group_8_match_df
        
    #     #TG(mg/dL)
    #     Group_9 = ALL_LAB.query(" h18 == \"09004C\"") #query order code
    #     Group_9['r2'] = Group_9['r2'] + "||" + " (" + Group_9['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_9 = Group_9.set_index(keys='id',drop=False)         #選定index
    #     Group_9 = pd.get_dummies(Group_9.r2)                      #one hot
    #     Group_9_list = list(Group_9)                              #取出r2 onehot包含哪些項
    #     Group_9_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_9_list)):
    #         insert_df9 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_9_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_9_list[i].replace('||','')
    #         insert_df9.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df9 = insert_df9[['id',column_name,'h11|h13']] #drop
    #         Group_9_match_df = pd.concat([Group_9_match_df,insert_df9],axis=0) #疊加
    #         i+=1
    #     Group_9_match_df

    #     #Albumin(g/dL)
    #     Group_10 = ALL_LAB.query(" h18 == \"09038C\"") #query order code
    #     Group_10['r2'] = Group_10['r2'] + "||" + " (" + Group_10['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_10 = Group_10.set_index(keys='id',drop=False)         #選定index
    #     Group_10 = pd.get_dummies(Group_10.r2)                      #one hot
    #     Group_10_list = list(Group_10)                              #取出r2 onehot包含哪些項
    #     Group_10_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_10_list)):
    #         insert_df10 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_10_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_10_list[i].replace('||','')
    #         insert_df10.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df10 = insert_df10[['id',column_name,'h11|h13']] #drop
    #         Group_10_match_df = pd.concat([Group_10_match_df,insert_df10],axis=0) #疊加
    #         i+=1
    #     Group_10_match_df

    #     #BUN(mg/dL)
    #     Group_11 = ALL_LAB.query(" h18 == \"09002C\"") #query order code
    #     Group_11['r2'] = Group_11['r2'] + "||" + " (" + Group_11['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_11 = Group_11.set_index(keys='id',drop=False)         #選定index
    #     Group_11 = pd.get_dummies(Group_11.r2)                      #one hot
    #     Group_11_list = list(Group_11)                              #取出r2 onehot包含哪些項
    #     Group_11_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_11_list)):
    #         insert_df11 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_11_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_11_list[i].replace('||','')
    #         insert_df11.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df11 = insert_df11[['id',column_name,'h11|h13']] #drop
    #         Group_11_match_df = pd.concat([Group_11_match_df,insert_df11],axis=0) #疊加
    #         i+=1
    #     Group_11_match_df

    #     #GFR(ml/min/1.73m^)
    #     Group_12 = ALL_LAB.query(" h18 == \"09015C\" | h18 == \"09016C\"") #query order code
    #     Group_12['r2'] = Group_12['r2'] + "||" + " (" + Group_12['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_12 = Group_12.set_index(keys='id',drop=False)         #選定index
    #     Group_12 = pd.get_dummies(Group_12.r2)                      #one hot
    #     Group_12_list = list(Group_12)                              #取出r2 onehot包含哪些項
    #     Group_12_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_12_list)):
    #         insert_df12 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_12_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_12_list[i].replace('||','')
    #         insert_df12.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df12 = insert_df12[['id',column_name,'h11|h13']] #drop
    #         Group_12_match_df = pd.concat([Group_12_match_df,insert_df12],axis=0) #疊加
    #         i+=1
    #     Group_12_match_df

    #     #Creatinine(urine)(mg/dL)
    #     Group_13 = ALL_LAB.query(" h18 == \"09016C\"") #query order code
    #     Group_13['r2'] = Group_13['r2'] + "||" + " (" + Group_13['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_13 = Group_13.set_index(keys='id',drop=False)         #選定index
    #     Group_13 = pd.get_dummies(Group_13.r2)                      #one hot
    #     Group_13_list = list(Group_13)                              #取出r2 onehot包含哪些項
    #     Group_13_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_13_list)):
    #         insert_df13 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_13_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_13_list[i].replace('||','')
    #         insert_df13.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df13 = insert_df13[['id',column_name,'h11|h13']] #drop
    #         Group_13_match_df = pd.concat([Group_13_match_df,insert_df13],axis=0) #疊加
    #         i+=1
    #     Group_13_match_df

    #     #PSA(ng/mL)
    #     Group_14 = ALL_LAB.query(" h18 == \"12081C\" | h18 == \"27052C\"") #query order code
    #     Group_14['r2'] = Group_14['r2'] + "||" + " (" + Group_14['r5'] + ")" #項目加單位作為旋轉後名稱
    #     Group_14 = Group_14.set_index(keys='id',drop=False)         #選定index
    #     Group_14 = pd.get_dummies(Group_14.r2)                      #one hot
    #     Group_14_list = list(Group_14)                              #取出r2 onehot包含哪些項
    #     Group_14_match_df = pd.DataFrame()                         #創建空欄位準備迴圈(直的直的跑)

    #     i=0
    #     for i in range(len(Group_14_list)):
    #         insert_df14 = ALL_LAB.melt(id_vars=['id','r2','r5','h18','h11|h13']).query("variable == \"r4\"").query(r2_to_command(Group_14_list[i].split('||')[0])) #切掉剛剛被加的單位，才能query到原本的字串
    #         column_name = Group_14_list[i].replace('||','')
    #         insert_df14.rename(columns={'value':column_name,'id':'id'}, inplace = True) #rename
    #         insert_df14 = insert_df14[['id',column_name,'h11|h13']] #drop
    #         Group_14_match_df = pd.concat([Group_14_match_df,insert_df14],axis=0) #疊加
    #         i+=1
    #     Group_14_match_df     

    #     dfs=[]
    #     if Group_1_match_df.empty == False :
    #         dfs.append(Group_1_match_df)
    #     if Group_2_match_df.empty == False :
    #         dfs.append(Group_2_match_df)
    #     if Group_3_match_df.empty == False :
    #         dfs.append(Group_3_match_df)
    #     if Group_4_match_df.empty == False :
    #         dfs.append(Group_4_match_df)
    #     if Group_5_match_df.empty == False :
    #         dfs.append(Group_5_match_df)
    #     if Group_6_match_df.empty == False :
    #         dfs.append(Group_6_match_df)
    #     if Group_7_match_df.empty == False :
    #         dfs.append(Group_7_match_df)
    #     if Group_8_match_df.empty == False :
    #         dfs.append(Group_8_match_df)
    #     if Group_9_match_df.empty == False :
    #         dfs.append(Group_9_match_df)
    #     if Group_10_match_df.empty == False :
    #         dfs.append(Group_10_match_df)
    #     if Group_11_match_df.empty == False :
    #         dfs.append(Group_11_match_df)
    #     if Group_12_match_df.empty == False :
    #         dfs.append(Group_12_match_df)
    #     if Group_13_match_df.empty == False :
    #         dfs.append(Group_13_match_df)
    #     if Group_14_match_df.empty == False :
    #         dfs.append(Group_14_match_df)          

    #     try:
    #         del [[Group_1_match_df, Group_2_match_df, Group_3_match_df, Group_4_match_df, Group_5_match_df, Group_6_match_df, Group_7_match_df, 
    #         Group_8_match_df, Group_9_match_df, Group_10_match_df, Group_11_match_df, Group_12_match_df, Group_13_match_df, Group_14_match_df]]
    #         gc.collect()
    #     except:
    #         pass

    #     df_final = ft.reduce(lambda left, right: pd.merge(left, right, on=['id','h11|h13'],how='outer',suffixes=('', '_delme')), dfs)
    #     df_final = df_final[[d for d in df_final.columns if not d.endswith('_delme')]]
    #     df_final = pd.merge(df_final, CRLF, on=['id'],how='left')
    #     df_final = df_final.drop_duplicates()
    #     # df_final = pd.merge(df_final, AE_row_count_before, on=['id'],how='left')
    #     # df_final = df_final.drop_duplicates()
    #     # df_final = pd.merge(df_final, BE_row_count_before, on=['id'],how='left',suffixes=('_AE', '_BE'))
    #     # df_final = df_final.drop_duplicates()
    #     df_final
        
    # except:
    #     df_final = pd.DataFrame({'no_match':[' no matching results']})
    df_final = pd.DataFrame({'stop':[' stop used']})
    ####################### recoding ##############################

    try:
        df_CRLF['age_group'] = df_CRLF['age'].apply(age_group)
    except:
        pass
    
    ###80%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('80%')
    f.close()
    ###80%進度點###

    ####################### demographic ##############################

    try:
        df_CRLF_demo = describe(df_CRLF).T
        df_CRLF_demo = missing(df_CRLF_demo, df_CRLF)
        df_CRLF_demo_c = split_v_c(df_CRLF_demo)[1] #[0]是跑連續數值
    except:
        df_CRLF_demo_c = pd.DataFrame({'no_match':['The data cannot be calculated demographic']})

    ###90%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('90%')
    f.close()
    ###90%進度點###

    ####################### Death ##############################

    df_DEATH = pd.read_sql("SELECT id,d2,d3,d4,d5,d6,d7 FROM " + "[" + "DEATH" + "]", conn)
    df_DEATH = pd.merge(df_CRLF_cohort, df_DEATH, how='inner', on=['id'], indicator=False)
    df_DEATH.drop('didiag',axis=1,inplace=True)
    if df_DEATH.empty == True:
        df_DEATH = pd.DataFrame({'no_match':['not found cohort']})

    conn.close()

    return(df_CRLF, AE_row_count_before, AE_row_count_after, BE_row_count_before, BE_row_count_after, 
           TOTFAO1_match_df, TOTFBO1_matchD_df, TOTFBO1_matchS_df,TOTFBO1_matchC_df, LABM1_Check_match_df,
           LABM1_Surgery_match_df, df_CASE_match_df,df_final,df_CRLF_demo_c,df_DEATH)

def C_CANCER_n(logic_structure):
    sort = logic_structure.get("table_select")
    i=0
    global table_list
    table_list = []
    for i in range(len(sort)):
        table_list.append(sort[i]['table'])
        i=i+1

    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)
    cursor = conn.cursor
    ######################### common function ############################    

    def cut_date(x):
        try:          
            if len(x)>8:
                x = x[0:8]
            x_f = x[0:4]
            x_f = x_f.replace('9999','1900')
            x_m = x[4:6]
            x_m = x_m.replace('99','01')
            x_b = x[6:8]
            x_b = x_b.replace('99','01')
            x = x_f + x_m + x_b
            x = parse(x)
            return x
        
        except:
            x_error = '19000101'
            x_error = parse(x_error)
            return x_error
    #----------------------------------------- 硬條件 ---------------------------------------------------
    disease = logic_structure['disease']    
    AE_num_exist = logic_structure['disease_table'].get('TOTFAE', None)
    BE_num_exist = logic_structure['disease_table'].get('TOTFBE', None)
    global df_TOTFAE_orig
    global df_TOTFBE_orig
    data ={'id':[''],'age':[''],'d9':[''],'d25':[''],'d10':[''],'gender':[''],'d19':[''],'d20':['']} #空表不能query 強迫宣告
    df_TOTFAE_orig = pd.DataFrame(data)
    df_TOTFBE_orig = pd.DataFrame(data)

    #write index path
    index_write = str(logic_structure['index_write'])
    plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer_n\\" + index_write +"\\config\\"

    ###0%進度點###
    process_path = plug_path+'Cn_process.txt'
    try:
        os.remove(process_path)
    except:
        pass
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('0%')
    f.close()
    ###0%進度點###

    #AE端
    if AE_num_exist is None:
        AE_ID = []

    else:
        df_TOTFAE = pd.read_sql("SELECT * FROM " + "TOTFAE", conn)
        df_TOTFAE['d11'] = df_TOTFAE['d11'].apply(cut_date)
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date)
        df_TOTFAE['age'] = (df_TOTFAE['d9'].dt.year) - (df_TOTFAE['d11'].dt.year)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE_orig = df_TOTFAE.copy(deep=True)

        i=0
        for each_query in range(len(logic_structure['disease_queryList'])):
            each = list(logic_structure['disease_queryList'].values())[i]
            df_TOTFAE = df_TOTFAE.query(each)
            i+=1

        def TOTFAE_diagcount(ICD):
                _19 = df_TOTFAE.loc[df_TOTFAE['d19'].str.startswith(ICD, na = False)] #各欄找ICD
                _20 = df_TOTFAE.loc[df_TOTFAE['d20'].str.startswith(ICD, na = False)]
                _21 = df_TOTFAE.loc[df_TOTFAE['d21'].str.startswith(ICD, na = False)]
                _22 = df_TOTFAE.loc[df_TOTFAE['d22'].str.startswith(ICD, na = False)]
                _23 = df_TOTFAE.loc[df_TOTFAE['d23'].str.startswith(ICD, na = False)]

                col_combine = pd.concat([_19,_20,_21,_22,_23])
                col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
                AE_row_diagcount = col_combine['id'].value_counts()                     #計算單人符合幾次
                AE_row_diagcount = AE_row_diagcount.to_frame()
                AE_row_diagcount = AE_row_diagcount.reset_index()
                AE_row_diagcount.rename(columns={'id': 'diagnose_count','index':'id'}, inplace=True)
                return AE_row_diagcount

        AE_row_diagcount = TOTFAE_diagcount(tuple(disease))
        AE_row_diagcount = AE_row_diagcount.query('diagnose_count >=' + str(AE_num_exist))
        AE_ID = AE_row_diagcount['id'].drop_duplicates().tolist()

    ###10%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('10%')
    f.close()
    ###10%進度點###

    #BE端
    if BE_num_exist is None:
        BE_ID = []

    else:
        df_TOTFBE = pd.read_sql("SELECT * FROM " + "TOTFBE", conn)
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date)
        df_TOTFBE['d6'] = df_TOTFBE['d6'].apply(cut_date)
        df_TOTFBE['age'] = (df_TOTFBE['d10'].dt.year) - (df_TOTFBE['d6'].dt.year)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE_orig = df_TOTFBE.copy(deep=True)

        i=0
        for each_query in range(len(logic_structure['disease_queryList'])):
            each = list(logic_structure['disease_queryList'].values())[i]
            df_TOTFBE = df_TOTFBE.query(each)
            i+=1

        def TOTFBE_diagcount(ICD):
                _25 = df_TOTFBE.loc[df_TOTFBE['d25'].str.startswith(ICD, na = False)] #各欄找ICD
                _26 = df_TOTFBE.loc[df_TOTFBE['d26'].str.startswith(ICD, na = False)]
                _27 = df_TOTFBE.loc[df_TOTFBE['d27'].str.startswith(ICD, na = False)]
                _28 = df_TOTFBE.loc[df_TOTFBE['d28'].str.startswith(ICD, na = False)]
                _29 = df_TOTFBE.loc[df_TOTFBE['d29'].str.startswith(ICD, na = False)]

                col_combine = pd.concat([_25,_26,_27,_28,_29])
                col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
                BE_row_diagcount = col_combine['id'].value_counts()                     #計算單人符合幾次
                BE_row_diagcount = BE_row_diagcount.to_frame()
                BE_row_diagcount = BE_row_diagcount.reset_index()
                BE_row_diagcount.rename(columns={'id': 'diagnose_count','index':'id'}, inplace=True)
                return BE_row_diagcount

        BE_row_diagcount = TOTFBE_diagcount(tuple(disease))
        BE_row_diagcount = BE_row_diagcount.query('diagnose_count >=' + str(BE_num_exist))
        BE_ID = BE_row_diagcount['id'].drop_duplicates().tolist()

    #intersece
    intersece_ID = set(AE_ID) | set(BE_ID)
    intersece_ID = list(intersece_ID)
    print(len(intersece_ID),len(AE_ID),len(BE_ID))
    
    if not intersece_ID:
        nomatch_df = pd.DataFrame({'no_match':['no matching any cohort']})
        ###100%進度點###
        process_path = plug_path+'Cn_process.txt'
        f = open(process_path, 'w')
        f.write('100%')
        f.close()
        ###100%進度點###
        return(nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, 
               nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df, nomatch_df)

    # 轉成db command語句 
    def ID_to_command(x):                            
        x = "id == " + '\"' + x + '\"'
        return(x)
    def AEquery_to_command(x):                            
        x = "d19 == " + '\"' + x + '\"' + ' | ' + "d20 == " + '\"' + x + '\"' + ' | ' + "d21 == " + '\"' + x + '\"' + ' | ' + "d22 == " + '\"' + x + '\"' + ' | ' + "d23 == " + '\"' + x + '\"'
        return(x)
    def BEquery_to_command(x):                            
        x = "d25 == " + '\"' + x + '\"' + ' | ' + "d26 == " + '\"' + x + '\"' + ' | ' + "d27 == " + '\"' + x + '\"' + ' | ' + "d28 == " + '\"' + x + '\"' + ' | ' + "d29 == " + '\"' + x + '\"'
        return(x)

    intersece_ID_c = list(map(ID_to_command, intersece_ID)) #類似 list版的 apply map to list 
    AE_c = list(map(AEquery_to_command, disease)) #類似 list版的 apply map to list 
    BE_c = list(map(BEquery_to_command, disease)) #類似 list版的 apply map to list
    
    #存活者 rowdata
    df_TOTFAE_query = pd.DataFrame()
    i=0
    for AE_queryID in range(len(intersece_ID_c)):
        insert_df = df_TOTFAE_orig.query(intersece_ID_c[i])
        df_TOTFAE_query = pd.concat([df_TOTFAE_query,insert_df],axis=0)
        df_TOTFAE_query = df_TOTFAE_query.drop_duplicates()
        i+=1
    df_TOTFAE_query 

    df_TOTFBE_query = pd.DataFrame()
    i=0
    for BE_queryID in range(len(intersece_ID_c)):
        insert_df = df_TOTFBE_orig.query(intersece_ID_c[i])
        df_TOTFBE_query = pd.concat([df_TOTFBE_query,insert_df],axis=0)
        df_TOTFBE_query = df_TOTFBE_query.drop_duplicates()
        i+=1 
    df_TOTFBE_query


    df_TOTFAE_query['d9|d10'] = df_TOTFAE_query['d9']
    df_TOTFBE_query['d9|d10'] = df_TOTFBE_query['d10']
    df_TOTFABE_query_ID = pd.concat([df_TOTFAE_query[['id','d9|d10']], df_TOTFBE_query[['id','d9|d10']]])
    df_TOTFABE_query_ID = df_TOTFABE_query_ID.drop_duplicates(subset=['id'])
    df_TOTFAE_query = df_TOTFAE_query.drop(['d9|d10'], axis=1) #AE
    df_TOTFBE_query = df_TOTFBE_query.drop(['d9|d10'], axis=1) #BE
    df_TOTFABE_query_ID # 存活者
    
    ###20%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('20%')
    f.close()
    ###20%進度點###

    # TOTFAE ########################## 正常篩選結果、無效回報
    def TOTFAE_count(Col_Name, ICD):
            _19 = df_TOTFAE.loc[df_TOTFAE['d19'].str.startswith((ICD), na = False)] #各欄找ICD
            _20 = df_TOTFAE.loc[df_TOTFAE['d20'].str.startswith((ICD), na = False)]
            _21 = df_TOTFAE.loc[df_TOTFAE['d21'].str.startswith((ICD), na = False)]
            _22 = df_TOTFAE.loc[df_TOTFAE['d22'].str.startswith((ICD), na = False)]
            _23 = df_TOTFAE.loc[df_TOTFAE['d23'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_19,_20,_21,_22,_23])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            AE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            AE_row_count = AE_row_count.to_frame()
            AE_row_count = AE_row_count.reset_index()
            AE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return AE_row_count

    #comobidity
    TOTFAE_index = table_list.index('TOTFAE')
    comobidity_num = len(logic_structure['table_select'][TOTFAE_index]['logic_before'])
    comobidity_num

    if comobidity_num < 1:          #config沒有按共病，故保持ABE清冊人   
        TOTFAE_ID = df_TOTFABE_query_ID[['id']] #存活ID冊(沒被篩)
        AE_row_count_before = pd.DataFrame({'id':['N/A'],'comobidity_N/A':['N/A']}) 

    else:
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_TOTFABE_query_ID, df_TOTFAE, how='left', on=['id'], indicator=False) #篩掉非ABE冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date)

        if df_TOTFAE.empty == True:
            AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE內 無同母群的人群']})

        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['d9|d10'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600

        # before
        df_TOTFAE_before = df_TOTFAE.query("diff_days <=" + str(start_time_TOTFAE) + " " + "and" + " " + "diff_days >=" + str(0))

        if df_TOTFAE_before.empty == True:
            AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity time no matching results']})

        else:
            comobidity_col_name = []    #收集config中的所有comobidity_name
            comobidity_condition =[]    #收集config中的所有comobidity_condition

            for c in range(comobidity_num):
                comobidity_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['col_name'])
                comobidity_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_before'][c]['condition'])

            c_ = 0
            for c_ in range(comobidity_num):
                if c_ == 0:
                    AE_row_count_before = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                else:
                    AE_row_count_before_append = TOTFAE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                    AE_row_count_before = pd.merge(AE_row_count_before, AE_row_count_before_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                c_+=1

            # d=0
            # for d in range(len(comobidity_col_name)):
            #     AE_row_count_before = AE_row_count_before.drop(AE_row_count_before[AE_row_count_before[comobidity_col_name[d]]<2].index) #刪除門診<2的人

            if AE_row_count_before.empty == True:
                AE_row_count_before = pd.DataFrame({'no_match':['TOTFAE Comorbidity count no matching results with >=2']})

            else:
                TOTFAE_ID_before = AE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                AE_row_count_before #輸出結果表

    AE_row_count_before

    #complication
    TOTFAE_index = table_list.index('TOTFAE')
    complication_num = len(logic_structure['table_select'][TOTFAE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFAE_ID = df_TOTFABE_query_ID[['id']] #存活ID冊(沒被篩)
        AE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFAE = pd.read_sql("SELECT d3,d9,d19,d20,d21,d22,d23,verify FROM " + "TOTFAE", conn)
        df_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFAE = pd.merge(df_TOTFABE_query_ID, df_TOTFAE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFAE['d9'] = df_TOTFAE['d9'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date

        if df_TOTFAE.empty == True:
            AE_row_count_after  = pd.DataFrame({'no_match':['TOTFAE內 無同癌登的人群']})

        else:
            df_TOTFAE['diff_days'] = df_TOTFAE['d9|d10'] - df_TOTFAE['d9'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFAE['diff_days'] = df_TOTFAE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFAE = int(logic_structure['table_select'][TOTFAE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFAE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFAE = -(int(logic_structure['table_select'][TOTFAE_index]['end_time'])*30)
            except:
                end_time_TOTFAE = -3600

            #after
            df_TOTFAE_after = df_TOTFAE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAE))

            if df_TOTFAE_after.empty == True:
                AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication time no matching results']})

            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition

                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFAE_index]['logic_after'][c]['condition'])

                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        AE_row_count_after = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        AE_row_count_after_append = TOTFAE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        AE_row_count_after = pd.merge(AE_row_count_after, AE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1

                d = 0
                # for d in range(len(complication_col_name)):
                #     AE_row_count_after = AE_row_count_after.drop(AE_row_count_after[AE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人

                if AE_row_count_after.empty == True:
                    AE_row_count_after = pd.DataFrame({'no_match':['TOTFAE complication count no matching results with >=2']})

                else:
                    TOTFAE_ID_after = AE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    AE_row_count_after #輸出結果表

    ###30%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('30%')
    f.close()
    ###30%進度點###

    # TOTFBE ########################## 正常篩選結果、無效回報
    def TOTFBE_count(Col_Name, ICD):
            _25 = df_TOTFBE.loc[df_TOTFBE['d25'].str.startswith((ICD), na = False)] #各欄找ICD
            _26 = df_TOTFBE.loc[df_TOTFBE['d26'].str.startswith((ICD), na = False)]
            _27 = df_TOTFBE.loc[df_TOTFBE['d27'].str.startswith((ICD), na = False)]
            _28 = df_TOTFBE.loc[df_TOTFBE['d28'].str.startswith((ICD), na = False)]
            _29 = df_TOTFBE.loc[df_TOTFBE['d29'].str.startswith((ICD), na = False)]

            col_combine = pd.concat([_25,_26,_27,_28,_29])
            col_combine = col_combine.drop_duplicates(subset = ['verify'])          #去除可能重複筆
            BE_row_count = col_combine['id'].value_counts()                         #計算單人符合幾次
            BE_row_count = BE_row_count.to_frame()
            BE_row_count = BE_row_count.reset_index()
            BE_row_count.rename(columns={'id': Col_Name,'index':'id'}, inplace=True)
            return BE_row_count

    #comobidity
    TOTFBE_index = table_list.index('TOTFBE')
    comobidity_num = len(logic_structure['table_select'][TOTFBE_index]['logic_before'])

    if comobidity_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_TOTFABE_query_ID[['id']] #存活ID冊(沒被篩)
        BE_row_count_before = pd.DataFrame({'id':['N/A'],'comorbidity_N/A':['N/A']})

    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_TOTFABE_query_ID, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date)

        if df_TOTFBE.empty == True:
            BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
            BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE & Cohort no matching results']})
        else:

            df_TOTFBE['diff_days'] = df_TOTFBE['d9|d10'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600

            #before
            df_TOTFBE_before = df_TOTFBE.query("diff_days <=" + str(start_time_TOTFBE) + " " + "and" + " " + "diff_days >=" + str(0))

            if df_TOTFBE_before.empty == True:
                BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity time no matching results']})
            else:
                comobidity_col_name = []    #收集config中的所有comobidity_name
                comobidity_condition =[]    #收集config中的所有comobidity_condition

                for c in range(comobidity_num):
                    comobidity_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['col_name'])
                    comobidity_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_before'][c]['condition'])
                print(comobidity_col_name)
                print(comobidity_condition)
                c_ = 0
                for c_ in range(comobidity_num):
                    if c_ == 0:
                        BE_row_count_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_append_before = TOTFBE_count(comobidity_col_name[c_], tuple(comobidity_condition[c_]))
                        BE_row_count_before = pd.merge(BE_row_count_before, BE_row_count_append_before, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1

                # d=0
                # for d in range(len(comobidity_col_name)):
                #     BE_row_count_before = BE_row_count_before.drop(BE_row_count_before[BE_row_count_before[comobidity_col_name[d]]<1].index) #刪除住院<1的人

                if BE_row_count_before.empty == True:
                    BE_row_count_before = pd.DataFrame({'no_match':['TOTFBE comorbidity count no matching results with >=1']})

                else:
                    TOTFBE_ID_before = BE_row_count_before['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_before #輸出結果表

    #complication
    TOTFBE_index = table_list.index('TOTFBE')
    complication_num = len(logic_structure['table_select'][TOTFBE_index]['logic_after'])
    if complication_num < 1:          #config沒有按共病，故保持癌燈清冊人
        TOTFBE_ID = df_TOTFABE_query_ID[['id']] #存活ID冊(沒被篩)
        BE_row_count_after = pd.DataFrame({'id':['N/A'],'complication_N/A':['N/A']}) 

    else:   
        #astype、diff_days
        df_TOTFBE = pd.read_sql("SELECT d3,d10,d25,d26,d27,d28,d29,verify FROM " + "TOTFBE", conn)
        df_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
        df_TOTFBE = pd.merge(df_TOTFABE_query_ID, df_TOTFBE, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數   
        df_TOTFBE['d10'] = df_TOTFBE['d10'].apply(cut_date) # 信良幫助解 要先merge 才可以 cut_date
        if df_TOTFBE.empty == True:
            BE_row_count_after  = pd.DataFrame({'no_match':['TOTFBE內 無同癌登的人群']})

        else:
            df_TOTFBE['diff_days'] = df_TOTFBE['d9|d10'] - df_TOTFBE['d10'] #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
            df_TOTFBE['diff_days'] = df_TOTFBE['diff_days'].dt.days.astype('int')

            try:
                start_time_TOTFBE = int(logic_structure['table_select'][TOTFBE_index]['start_time'])*30               #月*30
            except:
                start_time_TOTFBE = 3600                                                                              #default 3600days   
            try:
                end_time_TOTFBE = -(int(logic_structure['table_select'][TOTFBE_index]['end_time'])*30)
            except:
                end_time_TOTFBE = -3600

            #after
            df_TOTFBE_after = df_TOTFBE.query("diff_days <=" + str(0) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBE))

            if df_TOTFBE_after.empty == True:
                BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication time no matching results']})

            else:
                complication_col_name = []    #收集config中的所有comobidity_name
                complication_condition =[]    #收集config中的所有comobidity_condition

                for c in range(complication_num):
                    complication_col_name.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['col_name'])
                    complication_condition.append(logic_structure['table_select'][TOTFBE_index]['logic_after'][c]['condition'])
                print(complication_col_name)
                print(complication_condition)
                c_ = 0
                for c_ in range(complication_num):
                    if c_ == 0:
                        BE_row_count_after = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_])) #進function list to tuple
                    else:
                        BE_row_count_after_append = TOTFBE_count(complication_col_name[c_], tuple(complication_condition[c_]))
                        BE_row_count_after = pd.merge(BE_row_count_after, BE_row_count_after_append, how='outer', on=['id'], indicator=False).fillna(value=0)
                    c_+=1

                # d = 0
                # for d in range(len(complication_col_name)):
                #     BE_row_count_after = BE_row_count_after.drop(BE_row_count_after[BE_row_count_after[complication_col_name[d]]<2].index) #刪除門診<2的人

                if BE_row_count_after.empty == True:
                    BE_row_count_after = pd.DataFrame({'no_match':['TOTFBE complication count no matching results with >=2']})

                else:
                    TOTFBE_ID_after = BE_row_count_after['id'].drop_duplicates().tolist() #存活ID冊(被篩)
                    BE_row_count_after #輸出結果表
    
    ###40%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('40%')
    f.close()
    ###40%進度點###

    ######################### TOTFAO1 ###########################           
    TOTFAO1_index = table_list.index('TOTFAO1')
    df_TOTFAO1 = pd.read_sql("SELECT * FROM " + "TOTFAO1", conn)
    df_TOTFAO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFAO1 = pd.merge(df_TOTFABE_query_ID, df_TOTFAO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFAO1['d9'] = df_TOTFAO1['d9'].apply(cut_date)    
    if df_TOTFAO1.empty == True:
        TOTFAO1_ID = df_TOTFABE_query_ID[['id']]
        TOTFAO1_match_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p4':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p9':['N/A'],'p10':['N/A'],'p13':['N/A'],'p14':['N/A'],'p15':['N/A'],'p17':['N/A'],'d9':['N/A']})

    else:
        df_TOTFAO1['diff_days'] = df_TOTFAO1['d9|d10'] - df_TOTFAO1['d9']  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFAO1['diff_days'] = df_TOTFAO1['diff_days'].dt.days.astype('int')

        try:
            start_time_TOTFAO1 = int(logic_structure['table_select'][TOTFAO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFAO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFAO1 = -(int(logic_structure['table_select'][TOTFAO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFAO1 = -3600

        df_TOTFAO1 = df_TOTFAO1.query("diff_days <=" + str(start_time_TOTFAO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFAO1))#動時間還未進藥篩

        if df_TOTFAO1.empty == True:
            TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug time no matching results']})

        else:
            TOTFAO1_match_df = pd.DataFrame()
            query_TOTFAO1 = logic_structure['table_select'][TOTFAO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFAO1_batch = query_TOTFAO1.split('|')
            try:
                i=0
                for A in range(len(query_TOTFAO1_batch)):
                    insert_df = df_TOTFAO1.query(query_TOTFAO1_batch[i])
                    TOTFAO1_match_df = pd.concat([TOTFAO1_match_df, insert_df], axis=0)
                    TOTFAO1_match_df = TOTFAO1_match_df.drop_duplicates()
                    i+=1

                if TOTFAO1_match_df.empty == True:
                    TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})

                else:
                    #藥品細項清單
                    TOTFAO1_match_df

                    #存活ID冊
                    TOTFAO1_ID = TOTFAO1_match_df['id'].drop_duplicates().tolist()
            except:
                TOTFAO1_match_df = pd.DataFrame({'no_match':['TOTFAO1 Drug code no matching results']})
                TOTFAO1_ID = []
    
    ###50%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('50%')
    f.close()
    ###50%進度點###

     ######################### TOTFBO1 ###########################
    df_TOTFBO1_sql = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    #藥物
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_TOTFABE_query_ID, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_TOTFABE_query_ID[['id']]
        TOTFBO1_matchD_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 

    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['d9|d10'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')

        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_drug'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_drug'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩

        if df_TOTFBO1.empty == True:
            TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})

        else:     
            TOTFBO1_matchD_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_drug']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')
            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchD_df = pd.concat([TOTFBO1_matchD_df, insert_df], axis=0)
                    TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop_duplicates()
                    i+=1

                if TOTFBO1_matchD_df.empty == True:
                    TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})


                else:
                    #醫令細項清單
                    TOTFBO1_matchD_df    

                    #存活ID冊
                    TOTFBO1_ID_D = TOTFBO1_matchD_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_D
            except:
                TOTFBO1_matchD_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_D=[]

    #手術
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_TOTFABE_query_ID, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_TOTFABE_query_ID[['id']]
        TOTFBO1_matchS_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 

    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['d9|d10'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')

        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_surgery'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_surgery'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩

        if df_TOTFBO1.empty == True:
            TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})

        else:     
            TOTFBO1_matchS_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_surgery']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')
            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchS_df = pd.concat([TOTFBO1_matchS_df, insert_df], axis=0)
                    TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop_duplicates()
                    i+=1

                if TOTFBO1_matchS_df.empty == True:
                    TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})


                else:
                    #醫令細項清單
                    TOTFBO1_matchS_df    

                    #存活ID冊
                    TOTFBO1_ID_S = TOTFBO1_matchS_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_S
            except:
                TOTFBO1_matchS_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_S=[]

    #檢驗檢查
    TOTFBO1_index = table_list.index('TOTFBO1')
    df_TOTFBO1 = df_TOTFBO1_sql.copy()
    df_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    df_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    df_TOTFBO1 = pd.merge(df_TOTFABE_query_ID, df_TOTFBO1, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_TOTFBO1['d10'] = df_TOTFBO1['d10'].apply(cut_date)

    if df_TOTFBO1.empty == True:
        TOTFBO1_ID = df_TOTFABE_query_ID[['id']]
        TOTFBO1_matchC_df = pd.DataFrame({'id':['N/A'],'didiag,':['N/A'],'t2':['N/A'],'t3':['N/A'],'t5':['N/A'],'t6':['N/A'],'d1':['N/A'],'d2':['N/A'],'p1':['N/A'],'p2':['N/A'],'p3':['N/A'],'p5':['N/A'],'p6':['N/A'],'p7':['N/A'],'p8':['N/A'],'p10':['N/A'],'p14':['N/A'],'p15':['N/A'],'p16':['N/A'],'d10':['N/A']}) 

    else:
        df_TOTFBO1['diff_days'] = df_TOTFBO1['d9|d10'] - df_TOTFBO1['d10']                                            #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_TOTFBO1['diff_days'] = df_TOTFBO1['diff_days'].dt.days.astype('int')

        try:
            start_time_TOTFBO1 = int(logic_structure['table_select'][TOTFBO1_index]['start_time_check'])*30               #月*30
        except:
            start_time_TOTFBO1 = 3600                                                                              #default 3600days   
        try:
            end_time_TOTFBO1 = -(int(logic_structure['table_select'][TOTFBO1_index]['end_time_check'])*30)
        except:
            end_time_TOTFBO1 = -3600

        df_TOTFBO1 = df_TOTFBO1.query("diff_days <=" + str(start_time_TOTFBO1) + " " + "and" + " " + "diff_days >=" + str(end_time_TOTFBO1)) #動時間還未進藥篩

        if df_TOTFBO1.empty == True:
            TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order time no matching results']})

        else:     
            TOTFBO1_matchC_df = pd.DataFrame()
            query_TOTFBO1 = logic_structure['table_select'][TOTFBO1_index]['query_check']   #條件式太長會深度爆炸，故字串轉list逐步query，再塞進空df
            query_TOTFBO1_batch = query_TOTFBO1.split('|')
            
            try:
                i=0
                for B in range(len(query_TOTFBO1_batch)):
                    insert_df = df_TOTFBO1.query(query_TOTFBO1_batch[i])
                    TOTFBO1_matchC_df = pd.concat([TOTFBO1_matchC_df, insert_df], axis=0)
                    TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop_duplicates()
                    i+=1

                if TOTFBO1_matchC_df.empty == True:
                    TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})


                else:
                    #醫令細項清單
                    TOTFBO1_matchC_df    

                    #存活ID冊
                    TOTFBO1_ID_C = TOTFBO1_matchC_df['id'].drop_duplicates().tolist()
                    TOTFBO1_ID_C
            except:
                TOTFBO1_matchC_df = pd.DataFrame({'no_match':['TOTFBO1 Order code no matching results']})
                TOTFBO1_ID_C = []
                
                                                 
    ###60%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('60%')
    f.close()
    ###60%進度點###                                                  
    ######################### LABM1_Check ########################### 篩選結果存活ID冊、檢驗細項清單

    LABM1_Check_index = table_list.index('LABM1_Check')
    df_LABM1_Check = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11'] + df_LABM1_Check['h13']
    df_LABM1_Check.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Check = pd.merge(df_TOTFABE_query_ID, df_LABM1_Check, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Check['h11|h13'] = df_LABM1_Check['h11|h13'].apply(cut_date)

    if df_LABM1_Check.empty == True:
        LABM1_Check_ID = df_TOTFABE_query_ID[['id']]
        LABM1_Check_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A']}) 

    else:
        df_LABM1_Check['diff_days'] = df_LABM1_Check['d9|d10'] - df_LABM1_Check['h11|h13']  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Check['diff_days'] = df_LABM1_Check['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Check = int(logic_structure['table_select'][LABM1_Check_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Check = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Check = -(int(logic_structure['table_select'][LABM1_Check_index]['end_time'])*30)
        except:
            end_time_LABM1_Check = -3600

        df_LABM1_Check = df_LABM1_Check.query("diff_days <=" + str(start_time_LABM1_Check) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Check))

        if df_LABM1_Check.empty == True:
            LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check time no matching results']})

        else:
            LABM1_Check_match_df = pd.DataFrame()
            query_LABM1_Check = logic_structure['table_select'][LABM1_Check_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Check_batch = query_LABM1_Check.split('|')
            
            try:
                i=0
                for LC in range(len(query_LABM1_Check_batch)):
                    insert_df = df_LABM1_Check.query(query_LABM1_Check_batch[i])
                    LABM1_Check_match_df = pd.concat([LABM1_Check_match_df, insert_df], axis=0)
                    LABM1_Check_match_df = LABM1_Check_match_df.drop_duplicates()
                    i+=1

                if LABM1_Check_match_df.empty == True:
                    LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})

                else:
                    #檢驗細項清單
                    LABM1_Check_match_df

                    #存活ID冊
                    LABM1_Check_ID = LABM1_Check_match_df['id'].drop_duplicates().tolist()
                    LABM1_Check_ID
                    
            except:
                LABM1_Check_match_df = pd.DataFrame({'no_match':['LABM1 Check code no matching results']})
                LABM1_Check_ID=[]
   
    ######################### LABM1_Surgery ########################### 篩選結果存活ID冊、手術細項清單

    LABM1_Surgery_index = table_list.index('LABM1_Surgery')
    df_LABM1_Surgery = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11'] + df_LABM1_Surgery['h13']
    df_LABM1_Surgery.rename(columns = {'h9':'id'}, inplace = True)
    df_LABM1_Surgery = pd.merge(df_TOTFABE_query_ID, df_LABM1_Surgery, how='left', on=['id'], indicator=False).fillna(value='0') #篩掉非癌登ID冊者減少人數
    df_LABM1_Surgery['h11|h13'] = df_LABM1_Surgery['h11|h13'].apply(cut_date)

    if df_LABM1_Surgery.empty == True :
        LABM1_Surgery_ID = df_TOTFABE_query_ID[['id']]
        LABM1_Surgery_match_df = pd.DataFrame({'id':['N/A'],'didiag':['N/A'],'h1':['N/A'],'h2':['N/A'],'h3':['N/A'],'h4':['N/A'],'h5':['N/A'],'h6':['N/A'],'h7':['N/A'],'h8':['N/A'],'gender':['N/A'],'h10':['N/A'],'h11':['N/A'],'h12':['N/A'],'h13':['N/A'],'h14':['N/A'],'h17':['N/A'],'h18':['N/A'],'h22':['N/A'],'h23':['N/A'],'h25':['N/A'],'r1':['N/A'],'r2':['N/A'],'r3':['N/A'],'r4':['N/A'],'r5':['N/A'],'r6_1':['N/A'],'r6_2':['N/A'],'r7':['N/A'],'r8_1':['N/A'],'r10':['N/A'],'r12':['N/A']}) 
    else:
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['d9|d10'] - df_LABM1_Surgery['h11|h13']                                  #計算診斷日跟門診筆相差幾天(+過去 -代表未來)
        df_LABM1_Surgery['diff_days'] = df_LABM1_Surgery['diff_days'].dt.days.astype('int')

        try:
            start_time_LABM1_Surgery = int(logic_structure['table_select'][LABM1_Surgery_index]['start_time'])*30               #月*30
        except:
            start_time_LABM1_Surgery = 3600                                                                              #default 3600days   
        try:
            end_time_LABM1_Surgery = -(int(logic_structure['table_select'][LABM1_Surgery_index]['end_time'])*30)
        except:
            end_time_LABM1_Surgery = -3600

        df_LABM1_Surgery = df_LABM1_Surgery.query("diff_days <=" + str(start_time_LABM1_Surgery) + " " + "and" + " " + "diff_days >=" + str(end_time_LABM1_Surgery))

        if df_LABM1_Surgery.empty == True:
            LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery time no matching results']})

        else:

            LABM1_Surgery_match_df = pd.DataFrame()
            query_LABM1_Surgery = logic_structure['table_select'][LABM1_Surgery_index]['query']   #條件式太長會深度爆炸，故字串轉list逐步query
            query_LABM1_Surgery_batch = query_LABM1_Surgery.split('|')
            
            try:
                i=0
                for LS in range(len(query_LABM1_Surgery_batch)):
                    insert_df = df_LABM1_Surgery.query(query_LABM1_Surgery_batch[i])
                    LABM1_Surgery_match_df = pd.concat([LABM1_Surgery_match_df, insert_df], axis=0)
                    LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop_duplicates()
                    i+=1

                if LABM1_Surgery_match_df.empty == True:
                    LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})

                else:
                    #手術細項清單
                    LABM1_Surgery_match_df

                    #存活ID冊
                    LABM1_Surgery_ID = LABM1_Surgery_match_df['id'].drop_duplicates().tolist()
                    LABM1_Surgery_ID
            except:
                LABM1_Surgery_match_df = pd.DataFrame({'no_match':['LABM1 Surgery code no matching results']})
                LABM1_Surgery_ID=[]
    ######################### normal-output ############################
    try:
        df_TOTFAE_query = df_TOTFAE_query.drop(['Index','verify','CreateTime','ModifyTime','IsUploadHash'],axis=1)
    except:
        df_TOTFAE_query = df_TOTFAE_query
    
    try:
        df_TOTFBE_query = df_TOTFBE_query.drop(['Index','verify','CreateTime','ModifyTime','IsUploadHash'],axis=1)
    except:
        df_TOTFBE_query = df_TOTFBE_query
        
    try:
        AE_row_count_before = AE_row_count_before
    except:
        AE_row_count_before = AE_row_count_before
    
    try:
        AE_row_count_after = AE_row_count_after     
    except:
        AE_row_count_after = AE_row_count_after
    
    try:
        BE_row_count_before = BE_row_count_before   
    except:
        BE_row_count_before = BE_row_count_before

    try:
        BE_row_count_after = BE_row_count_after   
    except:
        BE_row_count_after = BE_row_count_after
        
    try:
        TOTFAO1_match_df = TOTFAO1_match_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFAO1_match_df = TOTFAO1_match_df
    
    try:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchD_df = TOTFBO1_matchD_df
    
    try:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchS_df = TOTFBO1_matchS_df
    
    try:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df.drop(['Index','verify','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        TOTFBO1_matchC_df = TOTFBO1_matchC_df
                
    try:
        LABM1_Check_match_df = LABM1_Check_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Check_match_df = LABM1_Check_match_df
        
    try:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime','diff_days'],axis=1)
    except:
        LABM1_Surgery_match_df = LABM1_Surgery_match_df
    
    ####################### AE+BE for demographic ##############################
    df_TOTFAE_js = df_TOTFAE_query[['id','gender','d9','d19','age']].drop_duplicates(subset=['id','d9','d19'])
    df_TOTFAE_js.rename(columns={'d9':'d9|d10'}, inplace = True)
    df_TOTFAE_js.rename(columns={'d19':'d19|d25'}, inplace = True)
    df_TOTFBE_js = df_TOTFBE_query[['id','gender','d10','d25','age']].drop_duplicates(subset=['id','d10','d25'])
    df_TOTFBE_js.rename(columns={'d10':'d9|d10'}, inplace = True)
    df_TOTFBE_js.rename(columns={'d25':'d19|d25'}, inplace = True)
    df_TOTFABE_js = pd.concat([df_TOTFAE_js, df_TOTFBE_js], axis=0)
    
    ####################### recoding ##############################
    try:
        df_TOTFABE_js['age_group'] = df_TOTFABE_js['age'].apply(age_group)
    except:
        pass

    ###80%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('80%')
    f.close()
    ###80%進度點### 
    ####################### demographic ##############################

    try:
        df_TOTFABE_js_demo = describe(df_TOTFABE_js).T
        df_TOTFABE_js_demo = missing(df_TOTFABE_js_demo, df_TOTFABE_js)
        df_TOTFABE_js_demo_c = split_v_c(df_TOTFABE_js_demo)[1] #[0]是跑連續數值
    except:
        df_TOTFABE_js_demo_c = pd.DataFrame({'no_match':['The data cannot be calculated demographic']})
    ####################### for plot ##################################
    
    df_TOTFABE_js_plot = df_TOTFABE_js[['id','gender','age_group','d19|d25']].drop_duplicates(subset = ["id"])

    ####################### keep ######################################
    try:
        keep_AE = logic_structure['keep_AE']
        df_TOTFAE_query = df_TOTFAE_query[keep_AE]
    except:
        df_TOTFAE_query = df_TOTFAE_query
        
    try:
        keep_BE = logic_structure['keep_BE']
        df_TOTFBE_query = df_TOTFBE_query[keep_BE]
    except:
        df_TOTFBE_query = df_TOTFBE_query

    ###100%進度點###
    process_path = plug_path+'Cn_process.txt'
    f = open(process_path, 'w')
    f.write('100%')
    f.close()
    ###100%進度點###

    ####################### Death ######################################
    df_DEATH = pd.read_sql("SELECT id,d2,d3,d4,d5,d6,d7 FROM " + "[" + "DEATH" + "]", conn)
    df_DEATH_query = pd.DataFrame()
    i=0
    for DEATH_queryID in range(len(intersece_ID_c)):
        insert_df = df_DEATH.query(intersece_ID_c[i])
        df_DEATH_query = pd.concat([df_DEATH_query,insert_df],axis=0)
        df_DEATH_query = df_DEATH_query.drop_duplicates()
        i+=1

    if df_DEATH_query.empty == True:
        df_DEATH_query = pd.DataFrame({'no_match':['not found cohort']})
    ####################### return ######################################
    
    conn.close()
    return(df_TOTFAE_query, df_TOTFBE_query, AE_row_count_before, AE_row_count_after, BE_row_count_before, 
           BE_row_count_after, TOTFAO1_match_df, TOTFBO1_matchD_df, TOTFBO1_matchS_df, TOTFBO1_matchC_df, 
           LABM1_Check_match_df, LABM1_Surgery_match_df,df_TOTFABE_js_demo_c,df_TOTFABE_js_plot,df_DEATH_query)

def Pack_zip(path):
    time_log = str(time.strftime("%Y-%m-%d-%H-%M-%S")) #抓現在時間
    startdir = ".\\tmp_json" #要壓縮的文件夾路徑
    file_news = startdir + time_log + ".zip" #壓縮後壓縮檔的名字
    z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) #壓縮套件
    for dirpath, dirnames, filenames in os.walk(startdir):  #文件夾 遍歷器
        fpath = dirpath.replace(startdir,'') #取代掉前面根目錄串，否則會從根目錄開始一路複製
        fpath = fpath and fpath + os.sep or '' #
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
            print(filename)
    z.close()
    print(file_news)
    print(path+file_news)
    shutil.move(file_news, path+file_news)

    return('packaging has been completed' + ' '+ path+file_news)

def disease_level():
    #db
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()
    
    #create fold
    today_date = time.strftime("%Y-%m-%d %H-%M-%S")

    path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\disease_level' 
    if not os.path.isdir(path_orig):
        os.mkdir(path_orig)

    path = path_orig+'\\'+today_date
    if not os.path.isdir(path):
        os.mkdir(path)

    ###0%進度點###
    process_path = 'disease_level_process.txt'
    try:
        os.remove(process_path)
    except:
        pass
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('0%')
    f.close()
    ###0%進度點###

    #Summary table
    Summary_0 = B_plus_form1('','')
    Summary_continuous = Summary_0[1].reset_index()
    Summary_categorical = Summary_0[2].reset_index()

    try:
        Summary_categorical_query=['id','dbirth']
        i=0
        for i in range(len(Summary_categorical_query)):
            Summary_categorical['top'] = np.where(Summary_categorical['index']==Summary_categorical_query[i], '', Summary_categorical['top'])
            Summary_categorical['freq'] = np.where(Summary_categorical['index']==Summary_categorical_query[i], '', Summary_categorical['freq'])
            i+=1
    except:
        pass

    # Summary_continuous.to_excel(path+'/Summary_continuous.xlsx', encoding='utf-8-sig', index = False) 
    # Summary_categorical.to_excel(path+'/Summary_categorical.xlsx', encoding='utf-8-sig', index = False)
    Summary_continuous = Summary_continuous.to_json(path+"\\Summary_continuous.json",orient='records',date_format = 'iso', double_precision=3) 
    Summary_categorical = Summary_categorical.to_json(path+"\\Summary_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    ###10%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('10%')
    f.close()
    ###10%進度點###

    #TOTFAE
    TOTFAE_0 = Bsingle_demo('TOTFAE','gender','d19','','','')
    TOTFAE_continuous = TOTFAE_0[0].reset_index()
    TOTFAE_categorical = TOTFAE_0[1].reset_index()

    TOTFAE_categorical_query=['t2','t3','t6','d2','d9','d10','d11','d3','d27','id']
    i=0
    for i in range(len(TOTFAE_categorical_query)):
        TOTFAE_categorical['top'] = np.where(TOTFAE_categorical['index']==TOTFAE_categorical_query[i], '', TOTFAE_categorical['top'])
        TOTFAE_categorical['freq'] = np.where(TOTFAE_categorical['index']==TOTFAE_categorical_query[i], '', TOTFAE_categorical['freq'])
        i+=1

    # TOTFAE_continuous.to_excel(path+'/TOTFAE_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFAE_categorical.to_excel(path+'/TOTFAE_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFAE_continuous = TOTFAE_continuous.to_json(path+"\\TOTFAE_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFAE_categorical = TOTFAE_categorical.to_json(path+"\\TOTFAE_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #TOTFBE
    TOTFBE_0 = Bsingle_demo('TOTFBE','gender','d25','','','')
    TOTFBE_continuous = TOTFBE_0[0].reset_index()
    TOTFBE_categorical = TOTFBE_0[1].reset_index()

    TOTFBE_categorical_query=['t2','t3','t6','d2','d3','d6','d10','d11','id']
    i=0
    for i in range(len(TOTFBE_categorical_query)):
        TOTFBE_categorical['top'] = np.where(TOTFBE_categorical['index']==TOTFBE_categorical_query[i], '', TOTFBE_categorical['top'])
        TOTFBE_categorical['freq'] = np.where(TOTFBE_categorical['index']==TOTFBE_categorical_query[i], '', TOTFBE_categorical['freq'])
        i+=1

    # TOTFBE_continuous.to_excel(path+'/TOTFBE_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFBE_categorical.to_excel(path+'/TOTFBE_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFBE_continuous = TOTFBE_continuous.to_json(path+"\\TOTFBE_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFBE_categorical = TOTFBE_categorical.to_json(path+"\\TOTFBE_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    # #LABD1
    # LABD1_0 = Bsingle_demo('LABD1','gender','h15','','','')
    # LABD1_continuous = LABD1_0[0].reset_index()
    # LABD1_categorical = LABD1_0[1].reset_index()


    # LABD1_categorical_query=['h2','h4','h5','h7','h8','h9','h10','h11','h12','h13','h14','h19','h20','h22','r1','r4','r5','r6-1','r6-2','r7','r10','r12','id']
    # i=0
    # for i in range(len(LABD1_categorical_query)):
    #     LABD1_categorical['top'] = np.where(LABD1_categorical['index']==LABD1_categorical_query[i], '', LABD1_categorical['top'])
    #     LABD1_categorical['freq'] = np.where(LABD1_categorical['index']==LABD1_categorical_query[i], '', LABD1_categorical['freq'])
    #     i+=1

    # LABD1_continuous.to_excel(path+'/LABD1_continuous.xlsx', encoding='utf-8-sig', index = False)
    # LABD1_categorical.to_excel(path+'/LABD1_categorical.xlsx', encoding='utf-8-sig', index = False)

    ###20%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('20%')
    f.close()
    ###20%進度點###

    #LABM1
    LABM1_0 = Bsingle_demo('LABM1','gender','h18','','','')
    LABM1_continuous = LABM1_0[0].reset_index()
    LABM1_categorical = LABM1_0[1].reset_index()

    LABM1_categorical_query=['h2','h4','h6','h8','h9','h10','h11','h12','h13','h14','h17','h22','h23','r1','r4','r5','r6-1','r6-2','r7','r10','r12','id']
    i=0
    for i in range(len(LABM1_categorical_query)):
        LABM1_categorical['top'] = np.where(LABM1_categorical['index']==LABM1_categorical_query[i], '', LABM1_categorical['top'])
        LABM1_categorical['freq'] = np.where(LABM1_categorical['index']==LABM1_categorical_query[i], '', LABM1_categorical['freq'])
        i+=1

    # LABM1_continuous.to_excel(path+'/LABM1_continuous.xlsx', encoding='utf-8-sig', index = False)
    # LABM1_categorical.to_excel(path+'/LABM1_categorical.xlsx', encoding='utf-8-sig', index = False)
    LABM1_continuous = LABM1_continuous.to_json(path+"\\LABM1_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    LABM1_categorical = LABM1_categorical.to_json(path+"\\LABM1_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    # #LABD2
    # LABD2_0 = Bsingle_demo('LABD2','gender','h15','','','')
    # LABD2_continuous = LABD2_0[0].reset_index()
    # LABD2_categorical = LABD2_0[1].reset_index()

    # LABD2_categorical_query=['h2','h4','h5','h7','h8','h9','h10','h11','h12','h13','h14','h19','h20','h22','r1','r4','r5','r6-1','r6-2','r7','r10','r12','id']
    # i=0
    # for i in range(len(LABD2_categorical_query)):
    #     LABD2_categorical['top'] = np.where(LABD2_categorical['index']==LABD2_categorical_query[i], '', LABD2_categorical['top'])
    #     LABD2_categorical['freq'] = np.where(LABD2_categorical['index']==LABD2_categorical_query[i], '', LABD2_categorical['freq'])
    #     i+=1

    # LABD2_continuous.to_excel(path+'/LABD2_continuous.xlsx', encoding='utf-8-sig', index = False)
    # LABD2_categorical.to_excel(path+'/LABD2_categorical.xlsx', encoding='utf-8-sig', index = False)

    #LABM2
    LABM2_0 = Bsingle_demo('LABM2','gender','h18','','','')
    LABM2_continuous = LABM2_0[0].reset_index()
    LABM2_categorical = LABM2_0[1].reset_index()

    LABM2_categorical_query=['h2','h4','h6','h8','h9','h10','h11','h12','h13','h14','h17','h22','h23','r1','r4','r5','r6-1','r6-2','r7','r10','r12','id']
    i=0
    for i in range(len(LABM2_categorical_query)):
        LABM2_categorical['top'] = np.where(LABM2_categorical['index']==LABM2_categorical_query[i], '', LABM2_categorical['top'])
        LABM2_categorical['freq'] = np.where(LABM2_categorical['index']==LABM2_categorical_query[i], '', LABM2_categorical['freq'])
        i+=1

    # LABM2_continuous.to_excel(path+'/LABM2_continuous.xlsx', encoding='utf-8-sig', index = False)
    # LABM2_categorical.to_excel(path+'/LABM2_categorical.xlsx', encoding='utf-8-sig', index = False)
    LABM2_continuous = LABM2_continuous.to_json(path+"\\LABM2_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    LABM2_categorical = LABM2_categorical.to_json(path+"\\LABM2_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    ###30%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('30%')
    f.close()
    ###30%進度點###

    #CRLF
    CRLF_0 = Bsingle_demo('CRLF','gender','site','','','')
    CRLF_continuous = CRLF_0[0].reset_index()
    CRLF_categorical = CRLF_0[1].reset_index()

    CRLF_categorical_query=['hospid','id','dbirth','resid','dcount','didiag','dmconf','dsdiag','dtrt_1st','dop_1st','dop_mds','drt_1st','drt_end','dsyt','dchem','dhorm','dimmu','dhtep','dtarget','dother','drecur','dlast']
    i=0
    for i in range(len(CRLF_categorical_query)):
        CRLF_categorical['top'] = np.where(CRLF_categorical['index']==CRLF_categorical_query[i], '', CRLF_categorical['top'])
        CRLF_categorical['freq'] = np.where(CRLF_categorical['index']==CRLF_categorical_query[i], '', CRLF_categorical['freq'])
        i+=1

    # CRLF_continuous.to_excel(path+'/CRLF_continuous.xlsx', encoding='utf-8-sig', index = False)
    # CRLF_categorical.to_excel(path+'/CRLF_categorical.xlsx', encoding='utf-8-sig', index = False)
    CRLF_continuous = CRLF_continuous.to_json(path+"\\CRLF_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    CRLF_categorical = CRLF_categorical.to_json(path+"\\CRLF_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    # #CRSF
    # CRSF_0 = Bsingle_demo('CRSF','gender','site','','','')
    # CRSF_continuous = CRSF_0[0].reset_index()
    # CRSF_categorical = CRSF_0[1].reset_index()

    # CRSF_categorical_query=['hospid','id','dbirth','resid','dcount','didiag','dmconf','dop_1st','drt_1st','dchem','dhorm','dimmu','dhtep','dtarget','dother']
    # i=0
    # for i in range(len(CRSF_categorical_query)):
    #     CRSF_categorical['top'] = np.where(CRSF_categorical['index']==CRSF_categorical_query[i], '', CRSF_categorical['top'])
    #     CRSF_categorical['freq'] = np.where(CRSF_categorical['index']==CRSF_categorical_query[i], '', CRSF_categorical['freq'])
    #     i+=1

    # CRSF_continuous.to_excel(path+'/CRSF_continuous.xlsx', encoding='utf-8-sig', index = False)
    # CRSF_categorical.to_excel(path+'/CRSF_categorical.xlsx', encoding='utf-8-sig', index = False)

    ###40%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('40%')
    f.close()
    ###40%進度點###

    #DEATH
    DEATH_0 = Bsingle_demo('DEATH','gender','d5','','','')
    DEATH_continuous = DEATH_0[0].reset_index()
    DEATH_categorical = DEATH_0[1].reset_index()

    DEATH_categorical_query=['id','d3','d4']
    i=0
    for i in range(len(DEATH_categorical_query)):
        DEATH_categorical['top'] = np.where(DEATH_categorical['index']==DEATH_categorical_query[i], '', DEATH_categorical['top'])
        DEATH_categorical['freq'] = np.where(DEATH_categorical['index']==DEATH_categorical_query[i], '', DEATH_categorical['freq'])
        i+=1

    # DEATH_continuous.to_excel(path+'/DEATH_continuous.xlsx', encoding='utf-8-sig', index = False)
    # DEATH_categorical.to_excel(path+'/DEATH_categorical.xlsx', encoding='utf-8-sig', index = False)
    DEATH_continuous = DEATH_continuous.to_json(path+"\\DEATH_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    DEATH_categorical = DEATH_categorical.to_json(path+"\\DEATH_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #CASE
    CASE_0 = Bsingle_demo('CASE','gender','m5','','','')
    CASE_continuous = CASE_0[0].reset_index()
    CASE_categorical = CASE_0[1].reset_index()

    CASE_categorical_query=['id','d3','m2','m3']
    i=0
    for i in range(len(CASE_categorical_query)):
        CASE_categorical['top'] = np.where(CASE_categorical['index']==CASE_categorical_query[i], '', CASE_categorical['top'])
        CASE_categorical['freq'] = np.where(CASE_categorical['index']==CASE_categorical_query[i], '', CASE_categorical['freq'])
        i+=1

    # CASE_continuous.to_excel(path+'/CASE_continuous.xlsx', encoding='utf-8-sig', index = False)
    # CASE_categorical.to_excel(path+'/CASE_categorical.xlsx', encoding='utf-8-sig', index = False)
    CASE_continuous = CASE_continuous.to_json(path+"\\CASE_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    CASE_categorical = CASE_categorical.to_json(path+"\\CASE_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #TOTFAO1
    TOTFAO1_0 = Bsingle_demo('TOTFAO1','p7','p4','','','')
    TOTFAO1_continuous = TOTFAO1_0[0].reset_index()
    TOTFAO1_categorical = TOTFAO1_0[1].reset_index()

    TOTFAO1_categorical_query=['p1','p5','p7','p10','p13','p14','p15','p17','id']
    i=0
    for i in range(len(TOTFAO1_categorical_query)):
        TOTFAO1_categorical['top'] = np.where(TOTFAO1_categorical['index']==TOTFAO1_categorical_query[i], '', TOTFAO1_categorical['top'])
        TOTFAO1_categorical['freq'] = np.where(TOTFAO1_categorical['index']==TOTFAO1_categorical_query[i], '', TOTFAO1_categorical['freq'])
        i+=1

    # TOTFAO1_continuous.to_excel(path+'/TOTFAO1_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFAO1_categorical.to_excel(path+'/TOTFAO1_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFAO1_continuous = TOTFAO1_continuous.to_json(path+"\\TOTFAO1_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFAO1_categorical = TOTFAO1_categorical.to_json(path+"\\TOTFAO1_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #TOTFBO1
    TOTFBO1_0 = Bsingle_demo('TOTFBO1','p8','p3','','','')
    TOTFBO1_continuous = TOTFBO1_0[0].reset_index()
    TOTFBO1_categorical = TOTFBO1_0[1].reset_index()

    TOTFBO1_categorical_query=['p1','p5','p6','p14','p15','p16','id']
    i=0
    for i in range(len(TOTFBO1_categorical_query)):
        TOTFBO1_categorical['top'] = np.where(TOTFBO1_categorical['index']==TOTFBO1_categorical_query[i], '', TOTFBO1_categorical['top'])
        TOTFBO1_categorical['freq'] = np.where(TOTFBO1_categorical['index']==TOTFBO1_categorical_query[i], '', TOTFBO1_categorical['freq'])
        i+=1

    # TOTFBO1_continuous.to_excel(path+'/TOTFBO1_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFBO1_categorical.to_excel(path+'/TOTFBO1_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFBO1_continuous = TOTFBO1_continuous.to_json(path+"\\TOTFBO1_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFBO1_categorical = TOTFBO1_categorical.to_json(path+"\\TOTFBO1_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #TOTFAO2
    TOTFAO2_0 = Bsingle_demo('TOTFAO2','p7','p4','','','')
    TOTFAO2_continuous = TOTFAO2_0[0].reset_index()
    TOTFAO2_categorical = TOTFAO2_0[1].reset_index()

    TOTFAO2_categorical_query=['p1','p5','p7','p10','p13','p14','p15','p17','id']
    i=0
    for i in range(len(TOTFAO2_categorical_query)):
        TOTFAO2_categorical['top'] = np.where(TOTFAO2_categorical['index']==TOTFAO2_categorical_query[i], '', TOTFAO2_categorical['top'])
        TOTFAO2_categorical['freq'] = np.where(TOTFAO2_categorical['index']==TOTFAO2_categorical_query[i], '', TOTFAO2_categorical['freq'])
        i+=1

    # TOTFAO2_continuous.to_excel(path+'/TOTFAO2_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFAO2_categorical.to_excel(path+'/TOTFAO2_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFAO2_continuous = TOTFAO2_continuous.to_json(path+"\\TOTFAO2_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFAO2_categorical = TOTFAO2_categorical.to_json(path+"\\TOTFAO2_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    #TOTFBO2
    TOTFBO2_0 = Bsingle_demo('TOTFBO2','p8','p3','','','')
    TOTFBO2_continuous = TOTFBO2_0[0].reset_index()
    TOTFBO2_categorical = TOTFBO2_0[1].reset_index()

    TOTFBO2_categorical_query=['p1','p5','p6','p14','p15','p16','id']
    for i in range(len(TOTFBO2_categorical_query)):
        TOTFBO2_categorical['top'] = np.where(TOTFBO2_categorical['index']==TOTFBO2_categorical_query[i], '', TOTFBO2_categorical['top'])
        TOTFBO2_categorical['freq'] = np.where(TOTFBO2_categorical['index']==TOTFBO2_categorical_query[i], '', TOTFBO2_categorical['freq'])
        i+=1

    # TOTFBO2_continuous.to_excel(path+'/TOTFBO2_continuous.xlsx', encoding='utf-8-sig', index = False)
    # TOTFBO2_categorical.to_excel(path+'/TOTFBO2_categorical.xlsx', encoding='utf-8-sig', index = False)
    TOTFBO2_continuous = TOTFBO2_continuous.to_json(path+"\\TOTFBO2_continuous.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFBO2_categorical = TOTFBO2_categorical.to_json(path+"\\TOTFBO2_categorical.json",orient='records',date_format = 'iso', double_precision=3)

    ##
    def detail_value_count_level(df,col):
        def full0(x):
            if len(x)<1:
                x = None
            return x

        df = df[[col]]
        freq = df.groupby([col]).count()
        freq1 = df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
        freq2 = df[col].value_counts()
        freq1=freq1.to_frame()
        freq2=freq2.to_frame()
        freq1.rename(columns = {col:col+'_1'}, inplace = True)
        freq2.rename(columns = {col:col+'_2'},inplace = True)
        res = pd.concat([freq1,freq2],axis=1)
        res = res.astype(str)
        res[col] = res[col+'_2'] +" "+ "("+  res[col+'_1'] + ")"
        res = res[[col]]
        res['code'] = res.index
        res.rename(columns = {col:"value_count"},inplace = True)
        res = pd.concat([res,freq2],axis=1)
        res.rename(columns = {col+'_2':"value"},inplace = True) 
        res = res[['code','value_count','value']]
        res = res.astype(str)
        res['code'] = res['code'].apply(full0)
        return(res)

    ###50%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('50%')
    f.close()
    ###50%進度點###

    #CRLF
    CRLF = pd.read_sql("SELECT id,site,sex,age FROM " + "[" + "CRLF" + "]" , conn)
    CRLF_mut = CRLF.copy()
    CRLF_mut = detail_value_count_level(CRLF_mut,'site')
    CRLF_dis = CRLF.drop_duplicates(subset = ['id','site'])
    CRLF_dis = detail_value_count_level(CRLF_dis,'site')
    CRLF_sex = CRLF.drop_duplicates(subset = ['id','sex'])
    CRLF_sex = detail_value_count_level(CRLF_sex,'sex')
    CRLF_age_group = CRLF.copy()
    try:
        CRLF_age_group['age_group'] = CRLF_age_group['age'].apply(age_group)
    except:
        CRLF_age_group['age_group'] = CRLF_age_group['age']
    CRLF_age_group = CRLF_age_group.drop_duplicates(subset = ['id','age_group'])
    CRLF_age_group = detail_value_count_level(CRLF_age_group,'age_group')

    # CRLF_mut.to_excel(path+'/CRLF_mut.xlsx', encoding='utf-8-sig', index = False)
    # CRLF_dis.to_excel(path+'/CRLF_dis.xlsx', encoding='utf-8-sig', index = False)
    CRLF_mut = CRLF_mut.to_json(path+"\\CRLF_mut.json",orient='records',date_format = 'iso', double_precision=3)
    CRLF_dis = CRLF_dis.to_json(path+"\\CRLF_dis.json",orient='records',date_format = 'iso', double_precision=3)
    CRLF_sex = CRLF_sex.to_json(path+"\\CRLF_sex.json",orient='records',date_format = 'iso', double_precision=3)
    CRLF_age_group = CRLF_age_group.to_json(path+"\\CRLF_age_group.json",orient='records',date_format = 'iso', double_precision=3)

    # #CRSF
    # CRSF = pd.read_sql("SELECT id,site FROM " + "[" + "CRSF" + "]" , conn)
    # CRSF_mut = CRSF.copy()
    # CRSF_mut = detail_value_count_level(CRSF_mut,'site')
    # CRSF_dis = CRSF.drop_duplicates(subset = ['id','site'])
    # CRSF_dis = detail_value_count_level(CRSF_dis,'site')
    # CRSF_mut.to_excel(path+'/CRSF_mut.xlsx', encoding='utf-8-sig', index = False)
    # CRSF_dis.to_excel(path+'/CRSF_dis.xlsx', encoding='utf-8-sig', index = False)

    ###60%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('60%')
    f.close()
    ###60%進度點###

    #TOTFAE
    TOTFAE = pd.read_sql("SELECT d3,d19,d20,d21,d22,d23,gender FROM " + "[" + "TOTFAE" + "]" , conn)
    TOTFAE_d19 = TOTFAE[['d3','d19']]
    TOTFAE_d19.rename(columns={'d19':'icd'}, inplace = True)
    TOTFAE_d20 = TOTFAE[['d3','d20']]
    TOTFAE_d20.rename(columns={'d20':'icd'}, inplace = True)
    TOTFAE_d21 = TOTFAE[['d3','d21']]
    TOTFAE_d21.rename(columns={'d21':'icd'}, inplace = True)
    TOTFAE_d22 = TOTFAE[['d3','d22']]
    TOTFAE_d22.rename(columns={'d22':'icd'}, inplace = True)
    TOTFAE_d23 = TOTFAE[['d3','d23']]
    TOTFAE_d23.rename(columns={'d23':'icd'}, inplace = True)
    TOTFAE = pd.concat([TOTFAE_d19,TOTFAE_d20,TOTFAE_d21,TOTFAE_d22,TOTFAE_d23])
    TOTFAE = TOTFAE.replace(r'^\s*$',np.nan,regex=True).dropna()
    TOTFAE_mut = TOTFAE.copy()
    TOTFAE_mut = detail_value_count_level(TOTFAE_mut,'icd')
    TOTFAE_dis = TOTFAE.drop_duplicates(subset = ['d3','icd'])
    TOTFAE_dis = detail_value_count_level(TOTFAE_dis,'icd')
    TOTFAE = pd.read_sql("SELECT d3,gender FROM " + "[" + "TOTFAE" + "]" , conn)
    TOTFAE_gender = TOTFAE.drop_duplicates(subset = ['d3','gender'])
    TOTFAE_gender = detail_value_count_level(TOTFAE_gender,'gender')

    # TOTFAE_mut.to_excel(path+'/TOTFAE_mut.xlsx', encoding='utf-8-sig', index = False)
    # TOTFAE_dis.to_excel(path+'/TOTFAE_dis.xlsx', encoding='utf-8-sig', index = False)
    TOTFAE_mut = TOTFAE_mut.to_json(path+"\\TOTFAE_mut.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFAE_dis = TOTFAE_dis.to_json(path+"\\TOTFAE_dis.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFAE_gender = TOTFAE_gender.to_json(path+"\\TOTFAE_gender.json",orient='records',date_format = 'iso', double_precision=3)

    ###70%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('70%')
    f.close()
    ###70%進度點###

    #TOTFBE
    TOTFBE = pd.read_sql("SELECT d3,d25,d26,d27,d28,d29,gender FROM " + "[" + "TOTFBE" + "]" , conn)
    TOTFBE_d25 = TOTFBE[['d3','d25']]
    TOTFBE_d25.rename(columns={'d25':'icd'}, inplace = True)
    TOTFBE_d26 = TOTFBE[['d3','d26']]
    TOTFBE_d26.rename(columns={'d26':'icd'}, inplace = True)
    TOTFBE_d27 = TOTFBE[['d3','d27']]
    TOTFBE_d27.rename(columns={'d27':'icd'}, inplace = True)
    TOTFBE_d28 = TOTFBE[['d3','d28']]
    TOTFBE_d28.rename(columns={'d28':'icd'}, inplace = True)
    TOTFBE_d29 = TOTFBE[['d3','d29']]
    TOTFBE_d29.rename(columns={'d29':'icd'}, inplace = True)
    TOTFBE = pd.concat([TOTFBE_d25,TOTFBE_d26,TOTFBE_d27,TOTFBE_d28,TOTFBE_d29])
    TOTFBE = TOTFBE.replace(r'^\s*$',np.nan,regex=True).dropna()
    TOTFBE_mut = TOTFBE.copy()
    TOTFBE_mut = detail_value_count_level(TOTFBE_mut,'icd')
    TOTFBE_dis = TOTFBE.drop_duplicates(subset = ['d3','icd'])
    TOTFBE_dis = detail_value_count_level(TOTFBE_dis,'icd')
    TOTFBE = pd.read_sql("SELECT d3,gender FROM " + "[" + "TOTFBE" + "]" , conn)
    TOTFBE_gender = TOTFBE.drop_duplicates(subset = ['d3','gender'])
    TOTFBE_gender = detail_value_count_level(TOTFBE_gender,'gender')

    # TOTFBE_mut.to_excel(path+'/TOTFBE_mut.xlsx', encoding='utf-8-sig', index = False)
    # TOTFBE_dis.to_excel(path+'/TOTFBE_dis.xlsx', encoding='utf-8-sig', index = False)
    TOTFBE_mut = TOTFBE_mut.to_json(path+"\\TOTFBE_mut.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFBE_dis = TOTFBE_dis.to_json(path+"\\TOTFBE_dis.json",orient='records',date_format = 'iso', double_precision=3)
    TOTFBE_gender = TOTFBE_gender.to_json(path+"\\TOTFBE_gender.json",orient='records',date_format = 'iso', double_precision=3)

    # DEATH
    DEATH = pd.read_sql("SELECT id,d5 FROM " + "[" + "DEATH" + "]" , conn)
    DEATH_d5 = DEATH.drop_duplicates(subset = ['id'])
    DEATH_d5 = detail_value_count_level(DEATH_d5,'d5')
    DEATH_d5 = DEATH_d5.to_json(path+"\\DEATH_d5.json",orient='records',date_format = 'iso', double_precision=3)

    ##################ACCESS#######################
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\Genes_access.db'
    conn = sqlite3.connect(sqldb)
    cursor = conn.cursor()

    def Bsingle_demo_access(dataframe):
        dataframe = dataframe.replace(r'^\s*$',np.nan,regex=True)
        des_dataframe = describe(dataframe).T
        des_dataframe  = missing(des_dataframe, dataframe)
        #des_dataframe_v  = split_v_c(des_dataframe)[0]
        #des_dataframe_v  = summ(des_dataframe_v.T,dataframe).T #時間數字代號等等會被字串sum
        #des_dataframe_c  = split_v_c(des_dataframe)[1]
        return(des_dataframe.reset_index())

    ###90%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('90%')
    f.close()
    ###90%進度點###

    #Patient_demographic
    df_Patient_demographic = pd.read_sql("SELECT Chart_no,FMI_no FROM " + "[" + 'Patient_demographic' + "]" , conn)
    df_Patient_demographic = Bsingle_demo_access(df_Patient_demographic)
    df_Patient_demographic = df_Patient_demographic[['index','count','unique','missing']]
    df_Patient_demographic = df_Patient_demographic.to_json(path+"\\D_access_Patient_demographic.json",orient='records',date_format = 'iso', double_precision=3)

    #Cancer_characteristic_newly_diagnosis
    df_Cancer_characteristic_newly_diagnosis = pd.read_sql("SELECT Chart_no,Diagnosis_status,Cancer_type,Sitemets1,Sitemets2,Sitemets3,Sitemets4,Sitemets5,Clinical_stage,Clinical_t,Clinical_n,Clinical_m,Pathologic_stage,Pathologic_t,Pathologic_n,Pathologic_m,AJCC_ed,Other_stage_system,Other_clinical_stage,Other_pathologic_stage FROM " + "[" + 'Cancer_characteristic_newly_diagnosis' + "]" , conn)
    df_Cancer_characteristic_newly_diagnosis = Bsingle_demo_access(df_Cancer_characteristic_newly_diagnosis)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_characteristic_newly_diagnosis['top'] = np.where(df_Cancer_characteristic_newly_diagnosis['index']==delete_item[i], '', df_Cancer_characteristic_newly_diagnosis['top'])
        df_Cancer_characteristic_newly_diagnosis['freq'] = np.where(df_Cancer_characteristic_newly_diagnosis['index']==delete_item[i], '', df_Cancer_characteristic_newly_diagnosis['freq'])
    df_Cancer_characteristic_newly_diagnosis = df_Cancer_characteristic_newly_diagnosis.to_json(path+"\\D_access_Cancer_characteristic_newly_diagnosis.json",orient='records',date_format = 'iso', double_precision=3)

    #-1 Cancer_type
    df_Cancer_characteristic_newly_diagnosis = pd.read_sql("SELECT Chart_no,Diagnosis_status,Cancer_type,Sitemets1,Sitemets2,Sitemets3,Sitemets4,Sitemets5,Clinical_stage,Clinical_t,Clinical_n,Clinical_m,Pathologic_stage,Pathologic_t,Pathologic_n,Pathologic_m,AJCC_ed,Other_stage_system,Other_clinical_stage,Other_pathologic_stage FROM " + "[" + 'Cancer_characteristic_newly_diagnosis' + "]" , conn)
    df_Cancer_characteristic_newly_diagnosis_decount = detail_value_count_level(df_Cancer_characteristic_newly_diagnosis,'Cancer_type')
    df_Cancer_characteristic_newly_diagnosis_decount = df_Cancer_characteristic_newly_diagnosis_decount.to_json(path+"\\D_access_Cancer_characteristic_newly_diagnosis_decount.json",orient='records',date_format = 'iso', double_precision=3)


    #Cancer_characteristic_recurrence
    df_Cancer_characteristic_recurrence = pd.read_sql("SELECT Chart_no,Diagnosis_status,Cancer_type,Initial_stage,Initial_TNM,Recur_date,Recur_stage,Recur_histology,Recur_behavior,Sitemets1,Sitemets2,Sitemets3,Sitemets4,Sitemets5 FROM " + "[" + 'Cancer_characteristic_recurrence' + "]" , conn)
    df_Cancer_characteristic_recurrence = Bsingle_demo_access(df_Cancer_characteristic_recurrence)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_characteristic_recurrence['top'] = np.where(df_Cancer_characteristic_recurrence['index']==delete_item[i], '', df_Cancer_characteristic_recurrence['top'])
        df_Cancer_characteristic_recurrence['freq'] = np.where(df_Cancer_characteristic_recurrence['index']==delete_item[i], '', df_Cancer_characteristic_recurrence['freq'])
    df_Cancer_characteristic_recurrence = df_Cancer_characteristic_recurrence.to_json(path+"\\D_access_Cancer_characteristic_recurrence.json",orient='records',date_format = 'iso', double_precision=3)


    #Performance_status
    df_Performance_status = pd.read_sql("SELECT Chart_no,Assess_date,PS_measure,ECOG_value,Karnofsky_value FROM " + "[" + 'Performance_status' + "]" , conn)
    df_Performance_status = Bsingle_demo_access(df_Performance_status)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Performance_status['top'] = np.where(df_Performance_status['index']==delete_item[i], '', df_Performance_status['top'])
        df_Performance_status['freq'] = np.where(df_Performance_status['index']==delete_item[i], '', df_Performance_status['freq'])
    df_Performance_status = df_Performance_status.to_json(path+"\\D_access_Performance_status.json",orient='records',date_format = 'iso', double_precision=3)

    #Stopping_reason_for_cancer_related_medication_inpatient
    df_Stopping_reason_for_cancer_related_medication_inpatient = pd.read_sql("SELECT Chart_no,Medication,Rx_end_date,Stop_reason FROM " + "[" + 'Stopping_reason_for_cancer_related_medication_inpatient' + "]" , conn)
    df_Stopping_reason_for_cancer_related_medication_inpatient = Bsingle_demo_access(df_Stopping_reason_for_cancer_related_medication_inpatient)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Stopping_reason_for_cancer_related_medication_inpatient['top'] = np.where(df_Stopping_reason_for_cancer_related_medication_inpatient['index']==delete_item[i], '', df_Stopping_reason_for_cancer_related_medication_inpatient['top'])
        df_Stopping_reason_for_cancer_related_medication_inpatient['freq'] = np.where(df_Stopping_reason_for_cancer_related_medication_inpatient['index']==delete_item[i], '', df_Stopping_reason_for_cancer_related_medication_inpatient['freq'])
    df_Stopping_reason_for_cancer_related_medication_inpatient = df_Stopping_reason_for_cancer_related_medication_inpatient.to_json(path+"\\D_access_Stopping_reason_for_cancer_related_medication_inpatient.json",orient='records',date_format = 'iso', double_precision=3)

    #-1 Medication
    df_Stopping_reason_for_cancer_related_medication_inpatient = pd.read_sql("SELECT Chart_no,Medication,Rx_end_date,Stop_reason FROM " + "[" + 'Stopping_reason_for_cancer_related_medication_inpatient' + "]" , conn)
    df_Stopping_reason_for_cancer_related_medication_inpatient_decount = detail_value_count_level(df_Stopping_reason_for_cancer_related_medication_inpatient,'Medication')
    df_Stopping_reason_for_cancer_related_medication_inpatient_decount = df_Stopping_reason_for_cancer_related_medication_inpatient_decount.to_json(path+"\\D_access_Stopping_reason_for_cancer_related_medication_inpatient_decount.json",orient='records',date_format = 'iso', double_precision=3)

    #Stopping_reason_for_cancer_related_medication_outpatient
    df_Stopping_reason_for_cancer_related_medication_outpatient = pd.read_sql("SELECT Chart_no,Medication,Rx_end_date,Stop_reason FROM " + "[" + 'Stopping_reason_for_cancer_related_medication_outpatient' + "]" , conn)
    df_Stopping_reason_for_cancer_related_medication_outpatient = Bsingle_demo_access(df_Stopping_reason_for_cancer_related_medication_outpatient)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Stopping_reason_for_cancer_related_medication_outpatient['top'] = np.where(df_Stopping_reason_for_cancer_related_medication_outpatient['index']==delete_item[i], '', df_Stopping_reason_for_cancer_related_medication_outpatient['top'])
        df_Stopping_reason_for_cancer_related_medication_outpatient['freq'] = np.where(df_Stopping_reason_for_cancer_related_medication_outpatient['index']==delete_item[i], '', df_Stopping_reason_for_cancer_related_medication_outpatient['freq'])
    df_Stopping_reason_for_cancer_related_medication_outpatient = df_Stopping_reason_for_cancer_related_medication_outpatient.to_json(path+"\\D_access_Stopping_reason_for_cancer_related_medication_outpatient.json",orient='records',date_format = 'iso', double_precision=3)

    #-1 Medication
    df_Stopping_reason_for_cancer_related_medication_outpatient = pd.read_sql("SELECT Chart_no,Medication,Rx_end_date,Stop_reason FROM " + "[" + 'Stopping_reason_for_cancer_related_medication_outpatient' + "]" , conn)
    df_Stopping_reason_for_cancer_related_medication_outpatient_decount = detail_value_count_level(df_Stopping_reason_for_cancer_related_medication_outpatient,'Medication')
    df_Stopping_reason_for_cancer_related_medication_outpatient_decount = df_Stopping_reason_for_cancer_related_medication_outpatient_decount.to_json(path+"\\D_access_Stopping_reason_for_cancer_related_medication_outpatient_decount.json",orient='records',date_format = 'iso', double_precision=3)

    #Cancer_related_radiotherapy
    df_Cancer_related_radiotherapy = pd.read_sql("SELECT Chart_no,RT_procedure,RT_site,ECOG_value,Karnofsky_value FROM " + "[" + 'Cancer_related_radiotherapy' + "]" , conn)
    df_Cancer_related_radiotherapy = Bsingle_demo_access(df_Cancer_related_radiotherapy)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_related_radiotherapy['top'] = np.where(df_Cancer_related_radiotherapy['index']==delete_item[i], '', df_Cancer_related_radiotherapy['top'])
        df_Cancer_related_radiotherapy['freq'] = np.where(df_Cancer_related_radiotherapy['index']==delete_item[i], '', df_Cancer_related_radiotherapy['freq'])
    df_Cancer_related_radiotherapy = df_Cancer_related_radiotherapy.to_json(path+"\\D_access_Cancer_related_radiotherapy.json",orient='records',date_format = 'iso', double_precision=3)

    ###95%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('95%')
    f.close()
    ###95%進度點###

    #-1 RT_site
    df_Cancer_related_radiotherapy = pd.read_sql("SELECT Chart_no,RT_procedure,RT_site,ECOG_value,Karnofsky_value FROM " + "[" + 'Cancer_related_radiotherapy' + "]" , conn)
    df_Cancer_related_radiotherapy_decount = detail_value_count_level(df_Cancer_related_radiotherapy,'RT_site')
    df_Cancer_related_radiotherapy_decount = df_Cancer_related_radiotherapy_decount.to_json(path+"\\D_access_Cancer_related_radiotherapy_decount.json",orient='records',date_format = 'iso', double_precision=3)

    #Cancer_treatment_outcome
    df_Cancer_treatment_outcome = pd.read_sql("SELECT Chart_no,Progression_date,Assess_criteria,ECOG_value,Karnofsky_value FROM " + "[" + 'Cancer_treatment_outcome' + "]" , conn)
    df_Cancer_treatment_outcome = Bsingle_demo_access(df_Cancer_treatment_outcome)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_treatment_outcome['top'] = np.where(df_Cancer_treatment_outcome['index']==delete_item[i], '', df_Cancer_treatment_outcome['top'])
        df_Cancer_treatment_outcome['freq'] = np.where(df_Cancer_treatment_outcome['index']==delete_item[i], '', df_Cancer_treatment_outcome['freq'])
    df_Cancer_treatment_outcome = df_Cancer_treatment_outcome.to_json(path+"\\D_access_Cancer_treatment_outcome.json",orient='records',date_format = 'iso', double_precision=3)

    #Follow_ups
    df_Follow_ups = pd.read_sql("SELECT Chart_no,Last_followup_date FROM " + "[" + 'Follow_ups' + "]" , conn)
    df_Follow_ups = Bsingle_demo_access(df_Follow_ups)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Follow_ups['top'] = np.where(df_Follow_ups['index']==delete_item[i], '', df_Follow_ups['top'])
        df_Follow_ups['freq'] = np.where(df_Follow_ups['index']==delete_item[i], '', df_Follow_ups['freq'])
    df_Follow_ups = df_Follow_ups.to_json(path+"\\D_access_Follow_ups.json",orient='records',date_format = 'iso', double_precision=3)

    #Value_indicator_of_the_project
    df_Value_indicator_of_the_project = pd.read_sql("SELECT Chart_no,Trial_enrol_date FROM " + "[" + 'Value_indicator_of_the_project' + "]" , conn)
    df_Value_indicator_of_the_project = Bsingle_demo_access(df_Value_indicator_of_the_project)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Value_indicator_of_the_project['top'] = np.where(df_Value_indicator_of_the_project['index']==delete_item[i], '', df_Value_indicator_of_the_project['top'])
        df_Value_indicator_of_the_project['freq'] = np.where(df_Value_indicator_of_the_project['index']==delete_item[i], '', df_Value_indicator_of_the_project['freq'])
    df_Value_indicator_of_the_project = df_Value_indicator_of_the_project.to_json(path+"\\D_access_Value_indicator_of_the_project.json",orient='records',date_format = 'iso', double_precision=3)

    #Cancer_related_medication_self_paid
    df_Cancer_related_medication_self_paid = pd.read_sql("SELECT Chart_no,Medication,Rx_start_date,Rx_end_date,Dosage,Dosage_unit,Frequency,Route,Stop_reason FROM " + "[" + 'Cancer_related_medication_self_paid' + "]" , conn)
    df_Cancer_related_medication_self_paid = Bsingle_demo_access(df_Cancer_related_medication_self_paid)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_related_medication_self_paid['top'] = np.where(df_Cancer_related_medication_self_paid['index']==delete_item[i], '', df_Cancer_related_medication_self_paid['top'])
        df_Cancer_related_medication_self_paid['freq'] = np.where(df_Cancer_related_medication_self_paid['index']==delete_item[i], '', df_Cancer_related_medication_self_paid['freq'])
    df_Cancer_related_medication_self_paid = df_Cancer_related_medication_self_paid.to_json(path+"\\D_access_Cancer_related_medication_self_paid.json",orient='records',date_format = 'iso', double_precision=3)

    #Cancer_related_medication_free_of_charge
    df_Cancer_related_medication_free_of_charge = pd.read_sql("SELECT Chart_no,Medication,Rx_start_date,Rx_end_date,Dosage,Dosage_unit,Frequency,Route,Stop_reason FROM " + "[" + 'Cancer_related_medication_free_of_charge' + "]" , conn)
    df_Cancer_related_medication_free_of_charge = Bsingle_demo_access(df_Cancer_related_medication_free_of_charge)
    delete_item=['Chart_no']
    for i in range(len(delete_item)):
        df_Cancer_related_medication_free_of_charge['top'] = np.where(df_Cancer_related_medication_free_of_charge['index']==delete_item[i], '', df_Cancer_related_medication_free_of_charge['top'])
        df_Cancer_related_medication_free_of_charge['freq'] = np.where(df_Cancer_related_medication_free_of_charge['index']==delete_item[i], '', df_Cancer_related_medication_free_of_charge['freq'])
    df_Cancer_related_medication_free_of_charge = df_Cancer_related_medication_free_of_charge.to_json(path+"\\D_access_Cancer_related_medication_free_of_charge.json",orient='records',date_format = 'iso', double_precision=3)

    #-1 Medication
    df_Cancer_related_medication_free_of_charge = pd.read_sql("SELECT Chart_no,Medication,Rx_start_date,Rx_end_date,Dosage,Dosage_unit,Frequency,Route,Stop_reason FROM " + "[" + 'Cancer_related_medication_free_of_charge' + "]" , conn)
    df_Cancer_related_medication_free_of_charge_decount = detail_value_count_level(df_Cancer_related_medication_free_of_charge,'Medication')
    df_Cancer_related_medication_free_of_charge_decount = df_Cancer_related_medication_free_of_charge_decount.to_json(path+"\\D_access_Cancer_related_medication_free_of_charge_decount.json",orient='records',date_format = 'iso', double_precision=3)

    conn.close()

    ###100%進度點###
    process_path = 'disease_level_process.txt'
    f = open(process_path, 'w')
    f.write('100%')
    f.close()
    ###100%進度點###

    return(path)

def json_to_excel(fileDir,left_columns):

    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()
    IdManagement = pd.read_sql("SELECT HashID" +','+ left_columns +' '+ "FROM IdManagement", conn)
    IdManagement.rename(columns = {'HashID':'id'}, inplace = True)
   
    fileExt = '.json'
    file_list_path = [os.path.join(fileDir, _) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    file_list_name = [_ for _ in os.listdir(fileDir) if _.endswith(fileExt)]
 
    i=0
    for i in range(len(file_list_path)):
        df_json = pd.read_json(file_list_path[i])

        try:
            df_json.drop('verify',axis=1,inplace=True)
        except:
            pass
        try:   
            df_json.drop('TableName',axis=1,inplace=True)
        except:
            pass
        try:
            df_json.drop('IsUploadHash',axis=1,inplace=True)
        except:
            pass
        try:
            df_json.drop('CreateTime',axis=1,inplace=True)
        except:
            pass
        try:
            df_json.drop('ModifyTime',axis=1,inplace=True)
        except:
            pass
        try:
            df_json.drop('Index',axis=1,inplace=True)
        except:
            pass
        try:
            df_json.drop('ModifyHASHTime',axis=1,inplace=True)
        except:
            pass

        try:
            df_json = pd.merge(df_json, IdManagement, how='left', on=['id'], indicator=False)
        except:
            pass

        df_json.to_excel(fileDir + "\\" + file_list_name[i].replace('.json','') + ".xlsx", index=False)

    return('OK')

def Version():
    Version = '1.1.5.230511'
    try:
        connT = str(sqlite3.connect("C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db"))
    except:
        connT = 'Connection failed'

    return(Version, connT)

def BiobankID_grouped(use_path):
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)
    cursor = conn.cursor()

    df_BiobankID = pd.read_sql("SELECT HashID,BiobankID FROM " + "BiobankManagement" , conn)
    df_BiobankID.rename(columns={'HashID':'id'}, inplace = True) 
    df_BiobankID = df_BiobankID.dropna()
    df_BiobankID_grouped = df_BiobankID.groupby('id')['BiobankID'].apply('|'.join).reset_index()
    df_BiobankID_grouped

    search_path = use_path+'\\*.json'

    file_list = glob.glob(search_path)
    if 'BiobankID' in str(file_list):
        return("already exists")
    else:
        for i in range(len(file_list)):
            try:
                each_df = pd.read_json(file_list[i], orient ='records', dtype=str)
                each_df_join = pd.merge(each_df, df_BiobankID_grouped, how='left', on=['id'], indicator=False)

                 #第一個從路徑切出檔名,第二個切掉副檔名
                each_df_json = each_df_join.to_json(use_path + "\\" + file_list[i].split('\\')[-1].split('.')[0] + "(BiobankID).json" ,orient='records',date_format = 'iso', double_precision=3)

            except:
                each_df_json = each_df.to_json(use_path + "\\" + file_list[i].split('\\')[-1].split('.')[0] + "(BiobankID).json" ,orient='records',date_format = 'iso', double_precision=3)

        return('OK')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False #json格式亂碼，改成ascii碼

@app.route("/Bsingle_demo", methods=['post'])
def post():
    global table,x,y,stats,ID,logic
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    query = query.split(',')
    table = query[0]
    x = query[1]
    y = query[2]
    stats = query[3]
    ID = query[4]
    logic = query[5]
    if table == 'CASE' and x == '' or table == 'case' and x == '':
        x = 'gender'
        y = 'm5'
    elif table == 'CRLF' and x == '' or table == 'crlf' and x == '' or table == 'CRSF' and x == '' or table == 'crsf' and x == '':
        x = 'gender'
        y = 'site'
    elif table == 'DEATH' and x == '' or table == 'death' and x == '':
        x = 'gender'
        y = 'd5'
    elif table == 'LABD1' and x == '' or table == 'labd1' and x == '':
        x = 'gender'
        y = 'h15'
    elif table == 'LABD2' and x == '' or table == 'labd2' and x == '':
        x = 'gender'
        y = 'h15'
    elif table == 'LABM1' and x == '' or table == 'labm1' and x == '':
        x = 'gender'
        y = 'h18'
    elif table == 'LABM2' and x == '' or table == 'labm2' and x == '':
        x = 'gender'
        y = 'h18'     
    elif table == 'TOTFAE' and x == '' or table == 'totfae' and x == '':
        x = 'gender'
        y = 'd19'
    elif table == 'TOTFAO1' and x == '' or table == 'totfao1' and x == '':
        x = 'p7'
        y = 'p4'
    elif table == 'TOTFAO2' and x == '' or table == 'totfao2' and x == '':
        x = 'p7'
        y = 'p4'
    elif table == 'TOTFBE' and x == '' or table == 'totfbe' and x == '':
        x = 'gender'
        y = 'd25'
    elif table == 'TOTFBO1' and x == '' or table == 'totfbo1' and x == '':
        x = 'p8'
        y = 'p3'
    elif table == 'TOTFBO2' and x == '' or table == 'totfbo2' and x == '':
        x = 'p8'
        y = 'p3'

    mix_sult_0 = Bsingle_demo(table,x,y,stats,ID,logic)

    if stats=='update':
        des_dataframe_update_v = mix_sult_0[0].reset_index()
        des_dataframe_update_c = mix_sult_0[1].reset_index()
        cross_dataframe_update = mix_sult_0[2].reset_index()
        dataframe_update = mix_sult_0[3]

    else:
        des_dataframe_v = mix_sult_0[0].reset_index()
        des_dataframe_c = mix_sult_0[1].reset_index()
        cross_dataframe = mix_sult_0[2].reset_index()
        dataframe = mix_sult_0[3]

    if stats=='update':
        des_dataframe_update_v = des_dataframe_update_v.to_json(orient='records',date_format = 'iso')
        des_dataframe_update_c = des_dataframe_update_c.to_json(orient='records',date_format = 'iso')
        cross_dataframe_update = cross_dataframe_update.to_json(orient='records',date_format = 'iso',double_precision=3)
        dataframe_update = dataframe_update.to_json(orient='records',date_format = 'iso')

    else:
        des_dataframe_v = des_dataframe_v.to_json(orient='records',date_format = 'iso')
        des_dataframe_c = des_dataframe_c.to_json(orient='records',date_format = 'iso')
        cross_dataframe = cross_dataframe.to_json(orient='records',date_format = 'iso',double_precision=3)
        dataframe = dataframe.to_json(orient='records',date_format = 'iso')

    
    if stats=='update':
        if len(ID) <=1:
            return jsonify(
            json.loads(des_dataframe_update_v),
            json.loads(des_dataframe_update_c),
            json.loads(cross_dataframe_update)
        )
        else:
            return jsonify(
            json.loads(des_dataframe_update_v),
            json.loads(des_dataframe_update_c),
            json.loads(dataframe_update)
        )

    else:
        if len(ID) <=1:
            return jsonify(
                json.loads(des_dataframe_v),
                json.loads(des_dataframe_c),
                json.loads(cross_dataframe)
        )
        else:
            return jsonify(
                json.loads(des_dataframe_v),
                json.loads(des_dataframe_c),
                json.loads(dataframe)
                
        )

@app.route("/Bsingle_demo", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Bnew_form2", methods=['post'])
def post2():
    global ID2,stats2
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    query = query.split(',')
    ID2 = query[0]
    stats2 = query[1]

    new_table_L = B_plus_form2(ID2,stats2)
    # des_new_table_L_v = B_plus_form2(ID2,stats2)[1].reset_index()
    # des_new_table_L_c = B_plus_form2(ID2,stats2)[2].reset_index()

    new_table_L = new_table_L.to_json(orient='records',date_format = 'iso', double_precision=3)
    # des_new_table_L_v = des_new_table_L_v.to_json(orient='records',date_format = 'iso', double_precision=3)
    # des_new_table_L_c = des_new_table_L_c.to_json(orient='records',date_format = 'iso', double_precision=3)
    return jsonify(
        json.loads(new_table_L)
        )
    # return jsonify(
    #     json.loads(new_table_L),
    #     json.loads(des_new_table_L_v),
    #     json.loads(des_new_table_L_c)
    #     )

@app.route("/Bnew_form2", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject2():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Bnew_form1", methods=['post'])
def post3():
    global stats3,logic3
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    query = query.split(',')
    stats3 = query[0]
    logic3 = query[1]

    mix_sult_3 = B_plus_form1(stats3,logic3)

    new_table_S = mix_sult_3[0]
    # if stats3 !='update':
    #     new_table_S.to_json("./Summary.json", orient = 'records',date_format = 'iso', double_precision=3)
    des_new_table_S_v  = mix_sult_3[1].reset_index()
    des_new_table_S_c  = mix_sult_3[2].reset_index()

    new_table_S = new_table_S.to_json(orient='records',date_format = 'iso', double_precision=3)
    des_new_table_S_v = des_new_table_S_v.to_json(orient='records',date_format = 'iso', double_precision=3)
    des_new_table_S_c = des_new_table_S_c.to_json(orient='records',date_format = 'iso', double_precision=3)
    return jsonify(
        json.loads(new_table_S),
        json.loads(des_new_table_S_v),
        json.loads(des_new_table_S_c)
        )

@app.route("/Bnew_form1", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject3():
    return  ("api refused connection, please try again with 'post'")

@app.route("/table_info", methods=['post'])
def post4():
    global table4,stats4,logic4
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    query = query.split(',')
    table4 = query[0]
    stats4 = query[1]
    logic4 = query[2]

    mix_sult_4 = table_info(table4,stats4,logic4)

    table_ = mix_sult_4[0]
    row = mix_sult_4[1]
    range_date = mix_sult_4[2]

    table_ = table_.to_json(orient='records',date_format = 'iso', double_precision=3)
    row = row.to_json(orient='records',date_format = 'iso', double_precision=3)
    range_date = range_date.to_json(orient='records',date_format = 'iso', double_precision=3)
    str_plus = "[" + table_ + "," + row + "," +  range_date + "]"  #因多不同表, 信良字串處理用

    return (str_plus
        )

@app.route("/table_info", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject4():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Bnew_form1_tmp", methods=['post'])
def post5():
    global logic5
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    logic5 = query

    mix_sult_5 = B_plus_form1_tmp(logic5)

    new_table_S = mix_sult_5[0]
    des_new_table_S_v  = mix_sult_5[1].reset_index()
    des_new_table_S_c  = mix_sult_5[2].reset_index()

    new_table_S = new_table_S.to_json(orient='records',date_format = 'iso', double_precision=3)
    des_new_table_S_v = des_new_table_S_v.to_json(orient='records',date_format = 'iso', double_precision=3)
    des_new_table_S_c = des_new_table_S_c.to_json(orient='records',date_format = 'iso', double_precision=3)
    return jsonify(
        json.loads(new_table_S),
        json.loads(des_new_table_S_v),
        json.loads(des_new_table_S_c)
        )

@app.route("/Bnew_form1_tmp", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject5():
    return  ("api refused connection, please try again with 'post'")

@app.route("/One_click_update", methods=['post'])
def post6():
    global last_update
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    last_update = query
    FINISH = One_click_update(last_update)
    B_plus_form1('','')
    return FINISH

@app.route("/One_click_update", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject6():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Bnew_form2_tmp", methods=['post'])
def post7():
    global ID7
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    ID7 = query
    with open('./tmp_json/'+ID7+'.json', newline='') as jsonfile:
            data = json.load(jsonfile)
    return jsonify(
        data
        )

@app.route("/Bnew_form2_tmp", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject7():
    return  ("api refused connection, please try again with 'post'")

@app.route("/progress", methods=['post','get'])
def post8():
    progress_sult = progress()
    return (
        progress_sult
        )

@app.route("/progress", methods=['put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject8():
    return  ("api refused connection, please try again with 'get'")

@app.route("/detail_value_count", methods=['post'])
def post9():
    global col
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    col = query
    js = detail_value_count(col)
    js = js.to_json(orient='records',date_format = 'iso', double_precision=3)

    return jsonify(
        json.loads(js)
        )

@app.route("/detail_value_count", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject9():
    return  ("api refused connection, please try again with 'get'")

@app.route("/C_CANCER", methods=['post'])
def post10():
    global logic_structure
    query = request.get_data()
    logic_structure = request.get_json()
    mix_sult_10 = C_CANCER(logic_structure)
    df_CRLF = mix_sult_10[0]
    AE_row_count_before = mix_sult_10[1]
    AE_row_count_after = mix_sult_10[2]
    BE_row_count_before = mix_sult_10[3]
    BE_row_count_after = mix_sult_10[4]
    TOTFAO1_match_df = mix_sult_10[5]
    TOTFBO1_matchD_df = mix_sult_10[6]
    TOTFBO1_matchS_df = mix_sult_10[7]
    TOTFBO1_matchC_df = mix_sult_10[8]
    LABM1_Check_match_df = mix_sult_10[9]
    LABM1_Surgery_match_df = mix_sult_10[10]
    df_CASE_match_df = mix_sult_10[11]
    df_final = mix_sult_10[12]
    df_CRLF_demo_c = mix_sult_10[13].reset_index()
    df_DEATH = mix_sult_10[14]

    df_CRLF = df_CRLF.to_json(orient='records',date_format = 'iso', double_precision=3)
    AE_row_count_before = AE_row_count_before.to_json(orient='records',date_format = 'iso', double_precision=3)
    AE_row_count_after = AE_row_count_after.to_json(orient='records',date_format = 'iso', double_precision=3)
    BE_row_count_before = BE_row_count_before.to_json(orient='records',date_format = 'iso', double_precision=3)
    BE_row_count_after = BE_row_count_after.to_json(orient='records',date_format = 'iso', double_precision=3)
    TOTFAO1_match_df = TOTFAO1_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    TOTFBO1_matchD_df = TOTFBO1_matchD_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    TOTFBO1_matchS_df = TOTFBO1_matchS_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    TOTFBO1_matchC_df = TOTFBO1_matchC_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    LABM1_Check_match_df = LABM1_Check_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    LABM1_Surgery_match_df = LABM1_Surgery_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    df_CASE_match_df = df_CASE_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
    df_final = df_final.to_json(orient='records',date_format = 'iso', double_precision=3)
    df_CRLF_demo_c = df_CRLF_demo_c.to_json(orient='records',date_format = 'iso', double_precision=3)
    df_DEATH = df_DEATH.to_json(orient='records',date_format = 'iso', double_precision=3)

    return jsonify(
        df_CRLF,
        AE_row_count_before,
        AE_row_count_after,
        BE_row_count_before,
        BE_row_count_after,
        TOTFAO1_match_df,
        TOTFBO1_matchD_df,
        TOTFBO1_matchS_df,
        TOTFBO1_matchC_df,
        LABM1_Check_match_df,
        LABM1_Surgery_match_df,
        df_CASE_match_df,
        df_final,
        df_CRLF_demo_c,
        df_DEATH
        )

@app.route("/C_CANCER", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject10():
    return  ("api refused connection, please try again with 'get'")

@app.route("/C_CANCER_n", methods=['post'])
def post11():
    global logic_structure
    query = request.get_data()
    logic_structure = request.get_json()
    mix_sult_11 = C_CANCER_n(logic_structure)
    df_TOTFAE_query = mix_sult_11[0]
    df_TOTFBE_query = mix_sult_11[1]
    AE_row_count_before = mix_sult_11[2]
    AE_row_count_after = mix_sult_11[3]
    BE_row_count_before = mix_sult_11[4]
    BE_row_count_after = mix_sult_11[5]
    TOTFAO1_match_df = mix_sult_11[6]
    TOTFBO1_matchD_df = mix_sult_11[7]
    TOTFBO1_matchS_df = mix_sult_11[8]
    TOTFBO1_matchC_df = mix_sult_11[9]
    LABM1_Check_match_df = mix_sult_11[10]
    LABM1_Surgery_match_df = mix_sult_11[11]
    df_TOTFABE_js_demo_c = mix_sult_11[12].reset_index()
    # df_TOTFABE_js_plot = mix_sult_11[13]
    df_DEATH_query = mix_sult_11[14]

    try:
        index_write = logic_structure['index_write']
    except:
        index_write=0
        
    if index_write>=1:
        #create fold
        create_date = time.strftime("%Y-%m-%d %H:%M:%S")

        path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer_n'
        if not os.path.isdir(path_orig):
            os.mkdir(path_orig)

        path = path_orig+'/'+str(index_write)
        if not os.path.isdir(path):
            os.mkdir(path)

        df_TOTFAE_query = df_TOTFAE_query.to_json(path+"\\Primary Table (TOTFAE).json",orient='records',date_format = 'iso', double_precision=3)
        df_TOTFBE_query = df_TOTFBE_query.to_json(path+"\\Primary Table (TOTFBE).json",orient='records',date_format = 'iso', double_precision=3)
        AE_row_count_before = AE_row_count_before.to_json(path+"\\Outpatient Comorbidity Table.json",orient='records',date_format = 'iso', double_precision=3)
        AE_row_count_after = AE_row_count_after.to_json(path+"\\Outpatient Complication Table.json",orient='records',date_format = 'iso', double_precision=3)
        BE_row_count_before = BE_row_count_before.to_json(path+"\\Inpatient Comorbidity Table.json",orient='records',date_format = 'iso', double_precision=3)
        BE_row_count_after = BE_row_count_after.to_json(path+"\\Inpatient Complication Table.json",orient='records',date_format = 'iso', double_precision=3)
        TOTFAO1_match_df = TOTFAO1_match_df.to_json(path+"\\Outpatient Details of Drug Table.json",orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchD_df = TOTFBO1_matchD_df.to_json(path+"\\Inpatient Details of Drug Table.json",orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchS_df = TOTFBO1_matchS_df.to_json(path+"\\Inpatient Details of Operation Table.json",orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchC_df = TOTFBO1_matchC_df.to_json(path+"\\Inpatient Details of Check Table.json",orient='records',date_format = 'iso', double_precision=3)
        LABM1_Check_match_df = LABM1_Check_match_df.to_json(path+"\\Laboratory Report Table (LAB).json",orient='records',date_format = 'iso', double_precision=3)
        LABM1_Surgery_match_df = LABM1_Surgery_match_df.to_json(path+"\\Operation Report Table (LAB).json",orient='records',date_format = 'iso', double_precision=3)
        df_TOTFABE_js_demo_c = df_TOTFABE_js_demo_c.to_json(path+"\\Description of Result.json",orient='records',date_format = 'iso', double_precision=3)
        # df_TOTFABE_js_plot = df_TOTFABE_js_plot.to_json(path+"\\df_TOTFABE_js_plot.json",orient='records',date_format = 'iso', double_precision=3)
        df_DEATH_query = df_DEATH_query.to_json(path+"\\DEATH.json",orient='records',date_format = 'iso', double_precision=3)

        def getdirsize(dir):
            size = 0
            for root, dirs, files in os.walk(dir):
                size += sum([getsize(join(root, name)) for name in files])
                
            return round(size/1024)

        size = getdirsize(path)

        #DB
        sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\NBCTDataDb.db'
        conn = sqlite3.connect(sqldb)
        cursor = conn.cursor()

        sql_update_query = """Update JsonData set ModifyTime =""" + "'" + create_date + "'" +','+"""Size ="""+"'"+ str(size) +"'"  + """where [Index] =""" +"'"+str(index_write)+"'"
        cursor.execute(sql_update_query)
        conn.commit()

        return("OK")

    else:
        df_TOTFAE_query = df_TOTFAE_query.to_json(orient='records',date_format = 'iso', double_precision=3)
        df_TOTFBE_query = df_TOTFBE_query.to_json(orient='records',date_format = 'iso', double_precision=3)
        AE_row_count_before = AE_row_count_before.to_json(orient='records',date_format = 'iso', double_precision=3)
        AE_row_count_after = AE_row_count_after.to_json(orient='records',date_format = 'iso', double_precision=3)
        BE_row_count_before = BE_row_count_before.to_json(orient='records',date_format = 'iso', double_precision=3)
        BE_row_count_after = BE_row_count_after.to_json(orient='records',date_format = 'iso', double_precision=3)
        TOTFAO1_match_df = TOTFAO1_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchD_df = TOTFBO1_matchD_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchS_df = TOTFBO1_matchS_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        TOTFBO1_matchC_df = TOTFBO1_matchC_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        LABM1_Check_match_df = LABM1_Check_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        LABM1_Surgery_match_df = LABM1_Surgery_match_df.to_json(orient='records',date_format = 'iso', double_precision=3)
        df_TOTFABE_js_demo_c = df_TOTFABE_js_demo_c.to_json(orient='records',date_format = 'iso', double_precision=3)
        # df_TOTFABE_js_plot = df_TOTFABE_js_plot.to_json(orient='records',date_format = 'iso', double_precision=3)
        df_DEATH_query = df_DEATH_query.to_json(orient='records',date_format = 'iso', double_precision=3)

        return jsonify(
            df_TOTFAE_query,
            df_TOTFBE_query,
            AE_row_count_before,
            AE_row_count_after,
            BE_row_count_before,
            BE_row_count_after,
            TOTFAO1_match_df,
            TOTFBO1_matchD_df,
            TOTFBO1_matchS_df,
            TOTFBO1_matchC_df,
            LABM1_Check_match_df,
            LABM1_Surgery_match_df,
            df_TOTFABE_js_demo_c,
            # df_TOTFABE_js_plot,
            df_DEATH_query
            )
        
@app.route("/C_CANCER_n", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject11():
    return  ("api refused connection, please try again with 'get'")

@app.route("/Pack_zip", methods=['post'])
def post12():
    global Pack_Path
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    Pack_Path = query
    mix_sult_12 = Pack_zip(Pack_Path)
    return(mix_sult_12)

@app.route("/Pack_zip", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject12():
    return  ("api refused connection, please try again with 'get'")

@app.route("/disease_level", methods=['post','get'])
def post13():
    mix_sult_13 = disease_level()
    return(mix_sult_13)

@app.route("/disease_level", methods=['put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject13():
    return  ("api refused connection, please try again with 'post or get'")

@app.route("/chart", methods=['post'])
def post14():
    global logic_structure
    query = request.get_data()
    logic_structure = request.get_json()
    mix_sult_10 = C_CANCER(logic_structure)
    df_CRLF = mix_sult_10[0]
    AE_row_count_before = mix_sult_10[1]
    # AE_row_count_after = mix_sult_10[2]
    BE_row_count_before = mix_sult_10[3]
    # BE_row_count_after = mix_sult_10[4]
    TOTFAO1_match_df = mix_sult_10[5]
    TOTFBO1_matchD_df = mix_sult_10[6]
    TOTFBO1_matchS_df = mix_sult_10[7]
    TOTFBO1_matchC_df = mix_sult_10[8]
    LABM1_Check_match_df = mix_sult_10[9]
    LABM1_Surgery_match_df = mix_sult_10[10]
    df_CASE_match_df = mix_sult_10[11]
    # df_final = mix_sult_10[12]
    # df_CRLF_demo_c = mix_sult_10[13].reset_index()
    df_DEATH = mix_sult_10[14]

    del mix_sult_10
    gc.collect()
    ###############################
    def date_removezero(x):
        if x == "0" or 0:
            return("")
        else:
            return(x)
    def spilt_8(s):
        try:
            s = str(s)
            return(s[0:8])
        except:
            s = str(s)
            return(s)
    ############Patient############
    cci=['Myocardial infarction','Congestive heart failure','Peripheral vascular disease','Cerebrovascular disease','Dementia','Chronic pulmonary disease','Rheumatic disease','Peptic ulcer disease','Mild liver disease','Diabetes without chronic complication','Diabetes with chronic complication','Hemiplegia or paraplegia','Renal disease','any malignancy','moderate or severe liver disease','Metastatic solid tumor','AIDS/HIV']
    def ps(df):

        def ps_(x):
            x=str(x)
            if x.isdigit()==True:
                return x[0:].zfill(3)
            
        def ps_Karnofsky(x):
            x=str(x)
            if x.isdigit()==True:
                if x == '100':             #100編碼的前兩碼是10會撞到 寫例外
                    return "100"
                if x == '1' or x == '001' or x == '2' or x == '002' or x == '3' or x == '003' or x == '4' or x == '004'or x == '5' or x == '005'or x == '6' or x == '006'or x == '7' or x == '007'or x == '8' or x == '008'or x == '9' or x == '009':
                    return "00"
                if x == '999' or x == '988':
                    return np.nan
                else:
                    return x[0:2].zfill(2) #0會不見補回來
            else:
                return np.nan
            
        def ps_ECOG(x):
            x=str(x)
            if x.isdigit()==True:
                if x[-1] == '0' or x[-1] == '1' or x[-1] == '2' or x[-1] == '3' or x[-1] == '4' or x[-1] == '5':
                    return x[-1]
                else:
                    return np.nan
            else:
                return np.nan

        df['ps_Karnofsky'] = df['ps'].apply(ps_Karnofsky)
        df['ps_ECOG'] = df['ps'].apply(ps_ECOG)
        df['ps'] = df['ps'].apply(ps_)
        return(df)

    def Comorbidity_bine(x):

        cc=0
        for cc in range(len(cci)):
            if  x[cci[cc]+"_x"]>=2 or x[cci[cc]+"_y"]>=1:
                x[cci[cc]+"_binary"] = 1
                cc+=1
            else:
                x[cci[cc]+"_binary"] = 0
                cc+=1
        return(x)

    #0508
    def ABE_fill_function(df):
        ABE_fill = df_CRLF[['id']].copy()
        ABE_fill['Myocardial infarction'] = 0
        ABE_fill['Congestive heart failure'] = 0
        ABE_fill['Peripheral vascular disease'] = 0
        ABE_fill['Cerebrovascular disease'] = 0
        ABE_fill['Dementia'] = 0
        ABE_fill['Chronic pulmonary disease'] = 0
        ABE_fill['Rheumatic disease'] = 0
        ABE_fill['Peptic ulcer disease'] = 0
        ABE_fill['Mild liver disease'] = 0
        ABE_fill['Diabetes without chronic complication'] = 0
        ABE_fill['Diabetes with chronic complication'] = 0
        ABE_fill['Hemiplegia or paraplegia'] = 0
        ABE_fill['Renal disease'] = 0
        ABE_fill['any malignancy'] = 0
        ABE_fill['moderate or severe liver disease'] = 0
        ABE_fill['Metastatic solid tumor'] = 0
        ABE_fill['AIDS/HIV'] = 0

        if 'no_match' in df.columns:
            df = ABE_fill

        return(df)

    Patient_output = pd.DataFrame()
    try:
        CRLF = df_CRLF
        CRLF_1 = CRLF.copy()
        CRLF_1 = CRLF_1[['hospid','id','sex','dbirth','resid','height','weight','smoking','btchew','drinking','ps']]
        CRLF_1['ps'] = CRLF_1['ps'].astype(str)

        #_x:AE _y:BE
        try:
            AE_row_count_before = ABE_fill_function(AE_row_count_before)
            BE_row_count_before = ABE_fill_function(BE_row_count_before)
            patient_Comorbidity = pd.merge(AE_row_count_before, BE_row_count_before, how='outer', on=['id'], indicator=False).fillna(np.nan)

            Patient_join = pd.merge(CRLF_1, patient_Comorbidity, how='outer', on=['id'], indicator=False).fillna(np.nan)
            Patient_output = Patient_join

        except:
            Patient_output = CRLF_1

        Patient_output = ps(Patient_output)
        Patient_output = Patient_output.apply(Comorbidity_bine, axis=1)
        
        d=0
        for d in range(len(cci)):
            try:
                Patient_output.drop([cci[d]+"_x", cci[d]+"_y"], axis=1, inplace=True)
            except:
                pass
            d+=1
    except:
        pass
    
    #04/24 RE
    try:
        Patient_output.drop('ps',axis=1,inplace=True)
        Patient_output.rename(columns = {'height':'Height_cm','weight':'Weight_kg','Myocardial infarction_binary':'com_1','Congestive heart failure_binary':'com_2','Peripheral vascular disease_binary':'com_3','Cerebrovascular disease_binary':'com_4','Dementia_binary':'com_5','Chronic pulmonary disease_binary':'com_6','Rheumatic disease_binary':'com_7','Peptic ulcer disease_binary':'com_8','Mild liver disease_binary':'com_9','Diabetes without chronic complication_binary':'com_10','Diabetes with chronic complication_binary':'com_11','Hemiplegia or paraplegia_binary':'com_12','Renal disease_binary':'com_13','any malignancy_binary':'com_14','moderate or severe liver disease_binary':'com_15','Metastatic solid tumor_binary':'com_16','AIDS/HIV_binary':'com_17'}, inplace = True)
    except:
        pass

    try:
        Patient_output['dbirth'] = Patient_output['dbirth'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass
    ###########Disease#############
    CRLF_2 = CRLF.copy()
    Disease_output = pd.DataFrame()
    try:
        Disease_output = CRLF_2[['id','age','sequence','class','class_d','class_t','dcont','didiag','site','lateral','hist','behavior','grade_c','grade_p','confirm','dmconf','size','pni','lvi','nexam','nposit','dsdiag','sdiag_o','sdiag_h','ct','cn','cm','cstage','cdescr','pt','pn','pm','pstage','pdescr','ajcc_ed','ostage','ostagec','ostagep','ssf1','ssf2','ssf3','ssf4','ssf5','ssf6','ssf7','ssf8','ssf9','ssf10']]
    except:
        pass

    #04/24 RE
    try:
        Disease_output.rename(columns = {'age':'di_age'}, inplace = True)
        Disease_output['di_age'] = Disease_output['di_age'].astype(str).str.zfill(2)

    except:
        pass

    try:
        Disease_output['dcont'] = Disease_output['dcont'].astype(str).str.replace('-','').apply(date_removezero)
        Disease_output['didiag'] = Disease_output['didiag'].astype(str).str.replace('-','').apply(date_removezero)
        Disease_output['dmconf'] = Disease_output['dmconf'].astype(str).str.replace('-','').apply(date_removezero)
        Disease_output['dsdiag'] = Disease_output['dsdiag'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass
    ###########Labs#############
    LAB_lab1 = pd.DataFrame()
    LAB_sug1 = pd.DataFrame()

    LAB_lab = LABM1_Check_match_df
    LAB_sug = LABM1_Surgery_match_df
    Labs_output = pd.DataFrame()

    try:
        LAB_lab1 = LAB_lab[['id','h18','h23','h25','r2','r3','r4','r5','r6_1','r6_2','r7','r8_1']]
    except:
        LAB_lab1
    try:
        LAB_sug1 = LAB_sug[['id','h18','h23','h25','r2','r3','r4','r5','r6_1','r6_2','r7','r8_1']]
    except:
        LAB_sug1
        
    Labs_output = pd.concat([Labs_output, LAB_lab1],axis=0)
    Labs_output = pd.concat([Labs_output, LAB_sug1],axis=0)

    #04/24 RE
    try:
        Labs_output.rename(columns = {'h18':'LAB_h18','h23':'LAB_h23','h25':'LAB_h25','r2':'LAB_r2','r3':'LAB_r3','r4':'LAB_r4','r5':'LAB_r5','r6_1':'LAB_r6_1','r6_2':'LAB_r6_2','r7':'LAB_r7','r8_1':'LAB_r8_1'}, inplace = True)
        
    except:
        pass

    try:
        Labs_output['LAB_h23'] = Labs_output['LAB_h23'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    #04/28 RE
    try:
        Labs_output['LAB_h23'] = Labs_output['LAB_h23'].apply(spilt_8)
    except:
        pass
    ###########Treatment_1#############
    CRLF_3 = CRLF.copy()
    Treatment_1_output = pd.DataFrame()
    try:
        Treatment_1_output = CRLF_3[['id','dtrt_1st','dop_1st','dop_mds','optype_o','optype_h','misurgery','smargin','smargin_d','opln_o','opln_h','opother_o','opother_h','noop','rtsumm','rtmodal','drt_1st','drt_end','srs','sls','rtstatus','ebrt','rth','rth_dose','rth_nf','rtl','rtl_dose','rtl_nf','ort_modal','ort_tech','ort','ort_dose','ort_nf','dsyt','chem_o','chem_h','dchem','horm_o','horm_h','dhorm','immu_o','immu_h','dimmu','htep_h','dhtep','target_o','target_h','dtarget','palli_h','other','dother']]
    except:
        pass

    try:
        Treatment_1_output['dtrt_1st'] = Treatment_1_output['dtrt_1st'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dop_1st'] = Treatment_1_output['dop_1st'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dop_mds'] = Treatment_1_output['dop_mds'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['drt_1st'] = Treatment_1_output['drt_1st'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['drt_end'] = Treatment_1_output['drt_end'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dchem'] = Treatment_1_output['dchem'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dhorm'] = Treatment_1_output['dhorm'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dimmu'] = Treatment_1_output['dimmu'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dhtep'] = Treatment_1_output['dhtep'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dtarget'] = Treatment_1_output['dtarget'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_1_output['dother'] = Treatment_1_output['dother'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass
    ###########Treatment_2#############
    #Drug >9 才為藥 ，從cancer fun來會被搜其他order_code故篩選排除
    Outpatient_Drug1 = pd.DataFrame()
    Outpatient_Drug = TOTFAO1_match_df
    try:
        Outpatient_Drug1 = Outpatient_Drug[['id','p4','p14','p15','p5','p1','p7','p9']]
        sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\drugs.db'
        conn = sqlite3.connect(sqldb)  
        cursor = conn.cursor()

        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p4'}, inplace = True)
        Outpatient_Drug1 = pd.merge(Outpatient_Drug1, ATC, how='left', on=['p4'], indicator=False)
        Outpatient_Drug1 = Outpatient_Drug1.loc[Outpatient_Drug1['p4'].str.len()==10]
    except:
        pass

    #04/24 RE
    try:
        Outpatient_Drug1.rename(columns = {'ATC_CODE':'atc_code_op','p4':'totfao1_p4','p14':'totfao1_p14','p15':'totfao1_p15','p5':'totfao1_p5','p1':'totfao1_p1','p7':'totfao1_p7','p9':'totfao1_p9'}, inplace = True)
    except:
        pass
    try:
        Outpatient_Drug1 = Outpatient_Drug1[['id','atc_code_op','totfao1_p4','totfao1_p14','totfao1_p15','totfao1_p5','totfao1_p1','totfao1_p7','totfao1_p9']]
    except:
        pass
    try:
        Outpatient_Drug1['totfao1_p14'] = Outpatient_Drug1['totfao1_p14'].astype(str).str.replace('-','').apply(date_removezero)
        Outpatient_Drug1['totfao1_p15'] = Outpatient_Drug1['totfao1_p15'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    #04/28 RE
    try:
        Outpatient_Drug1['totfao1_p14'] = Outpatient_Drug1['totfao1_p14'].apply(spilt_8)
        Outpatient_Drug1['totfao1_p15'] = Outpatient_Drug1['totfao1_p15'].apply(spilt_8)
    except:
        pass
    ###########Treatment_3#############
    Inpatient_Drug1 = pd.DataFrame()
    Inpatient_Drug = TOTFBO1_matchD_df

    try:
        Inpatient_Drug1 = Inpatient_Drug[['id','p3','p14','p15','p5','p1','p6','p7']]
        sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\drugs.db'
        conn = sqlite3.connect(sqldb)  
        cursor = conn.cursor()

        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p3'}, inplace = True)
        Inpatient_Drug1 = pd.merge(Inpatient_Drug1, ATC, how='left', on=['p3'], indicator=False)
        Inpatient_Drug1 = Inpatient_Drug1.loc[Inpatient_Drug1['p3'].str.len()==10]
    except:
        pass

    #04/24 RE
    try:
        Inpatient_Drug1.rename(columns = {'ATC_CODE':'atc_code_ip','p3':'totfbo1_p3','p14':'totfbo1_p14','p15':'totfbo1_p15','p5':'totfbo1_p5','p1':'totfbo1_p1','p6':'totfbo1_p6','p7':'totfbo1_p7'}, inplace = True)
    except:
        pass
    try:
        Inpatient_Drug1 = Inpatient_Drug1[['id','atc_code_ip','totfbo1_p3','totfbo1_p14','totfbo1_p15','totfbo1_p5','totfbo1_p1','totfbo1_p6','totfbo1_p7']]
    except:
        pass
    try:
        Inpatient_Drug1['totfbo1_p14'] = Inpatient_Drug1['totfbo1_p14'].astype(str).str.replace('-','').apply(date_removezero)
        Inpatient_Drug1['totfbo1_p15'] = Inpatient_Drug1['totfbo1_p15'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    #04/28 RE
    try:
        Inpatient_Drug1['totfbo1_p14'] = Inpatient_Drug1['totfbo1_p14'].apply(spilt_8)
        Inpatient_Drug1['totfbo1_p15'] = Inpatient_Drug1['totfbo1_p15'].apply(spilt_8)
    except:
        pass
    ###########Treatment_4#############
    # < 9 才為術檢碼 ，從cancer fun來會含十碼藥，這邊不要故排除
    Inpatient_check1 = pd.DataFrame()
    Inpatient_sur1 = pd.DataFrame()
    Outpatient_drug1 = pd.DataFrame()
    Inpatient_check = TOTFBO1_matchC_df
    Inpatient_sur = TOTFBO1_matchS_df
    Outpatient_drug = TOTFAO1_match_df
    Treatment_4_output = pd.DataFrame()

    try:
        Inpatient_check1 = Inpatient_check[['id','p3','p10','p14','p15']]
    except:
        Inpatient_check1
    try:
        Inpatient_sur1 = Inpatient_sur[['id','p3','p10','p14','p15']]
    except:
        Inpatient_sur1
    try:
        Outpatient_drug1 = Outpatient_drug[['id','p4','p6','p14','p15']]
        Outpatient_drug1 = Outpatient_drug1.loc[Outpatient_drug1['p4'].str.len()<10]
        Outpatient_drug1 = Outpatient_drug1.loc[Outpatient_drug1['p4'].str.len()>=6]
    except:
        Outpatient_drug1
    
    Treatment_4_output = pd.concat([Treatment_4_output, Inpatient_check1],axis=0)
    Treatment_4_output = pd.concat([Treatment_4_output, Inpatient_sur1],axis=0)
    Treatment_4_output = pd.concat([Treatment_4_output, Outpatient_drug1],axis=0)

    #04/24 RE
    try:
        Treatment_4_output.rename(columns = {'p4':'totfao1_p4','p3':'totfbo1_p3','p6':'totfao1_p6','p10':'totfbo1_p10','p14':'totfabo1_p14','p15':'totfabo1_p15'}, inplace = True)
    except:
        pass
    try:
        Treatment_4_output = Treatment_4_output[['id','totfao1_p4','totfbo1_p3','totfao1_p6','totfbo1_p10','totfabo1_p14','totfabo1_p15']]
    except:
        pass
    try:
        Treatment_4_output['totfabo1_p14'] = Treatment_4_output['totfabo1_p14'].astype(str).str.replace('-','').apply(date_removezero)
        Treatment_4_output['totfabo1_p15'] = Treatment_4_output['totfabo1_p15'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    #04/28 RE
    try:
        Treatment_4_output['totfabo1_p14'] = Treatment_4_output['totfabo1_p14'].apply(spilt_8)
        Treatment_4_output['totfabo1_p15'] = Treatment_4_output['totfabo1_p15'].apply(spilt_8)
    except:
        pass
    ###########Outcome#############
    def vstatus2(x):

        if x['vstatus'] ==0 or x['vstatus'] =='0':
            x['vstatus2'] = '0'
            
        elif len(str(x['d4'])) >= 4:
            x['vstatus2'] = '0'
            
        elif x['m7'] ==1 or x['m7'] ==2 or x['m7'] ==3 or x['m7'] =='1' or x['m7'] =='2' or x['m7'] =='3':
            x['vstatus2'] = '0'
            
        else:
            x['vstatus2'] = '1'
            
        return x
      
    CRLF_4 = CRLF.copy()
    CRLF1 = pd.DataFrame()
    DEATH1 = pd.DataFrame({'id': None, 'd4': None, 'd5': None},index=[0])
    CASE1 = pd.DataFrame({'id': None, 'm3': None, 'm5': None, 'm6': None, 'm7': None},index=[0])
    DEATH = df_DEATH
    CASE = df_CASE_match_df
    Outcome_output = pd.DataFrame()

    try:
        CRLF_1 = CRLF_4[['id','drecur','recur','dlast','vstatus']]
    except:
        CRLF_1
    try:
        DEATH1 = DEATH[['id','d4','d5']]
    except:
        DEATH1
    try:
        CASE1 =  CASE[['id','m3','m5','m6','m7']]
    except:
        CASE1

    try:
        Outcome_output = pd.merge(CRLF_1, DEATH1, how='left', on=['id'], indicator=False).fillna(np.nan)
        Outcome_output = pd.merge(Outcome_output, CASE1, how='left', on=['id'], indicator=False).fillna(np.nan)
        Outcome_output = Outcome_output.apply(vstatus2, axis=1)
    except:
        Outcome_output = pd.DataFrame()

    #04/24 RE
    try:
        Outcome_output.rename(columns = {'d4':'death_d4','d5':'death_d5','m3':'case_m3','m5':'case_m5','m6':'case_m6','m7':'case_m7','drecur':'crlf_drecur','vstatus':'vstatus_1','vstatus2':'vstatus_2'}, inplace = True)
    except:
        pass
    try:
        Outcome_output = Outcome_output[['id','death_d4','death_d5','case_m3','case_m5','case_m6','case_m7','crlf_drecur','recur','dlast','vstatus_1','vstatus_2']]
    except:
        pass
    try:
        Outcome_output['death_d4'] = Outcome_output['death_d4'].astype(str).str.replace('-','').apply(date_removezero)
        Outcome_output['case_m3'] = Outcome_output['case_m3'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ########### Null table fill #######
    def Null_table(df):
        if df.empty == True:
            df = pd.DataFrame({'no_match':['not found cohort']})
            return(df)
        else:
            df = df.drop_duplicates()
            df = df.replace("nan","")
            return(df)
    Patient_output = Null_table(Patient_output)
    Disease_output = Null_table(Disease_output)
    Labs_output = Null_table(Labs_output)
    Treatment_1_output = Null_table(Treatment_1_output)
    Outpatient_Drug1 = Null_table(Outpatient_Drug1)
    Inpatient_Drug1 = Null_table(Inpatient_Drug1)
    Treatment_4_output = Null_table(Treatment_4_output)
    Outcome_output = Null_table(Outcome_output)

    ###########drop N/A################
    try:
        Patient_output.drop(Patient_output[(Patient_output['id'] == 'N/A')].index, inplace=True)
    except:
        pass

    #write index path
    index_write = str(logic_structure['index_write'])
    is_coop = str(logic_structure['coop'])
    if is_coop == "0":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer\\" + index_write +"\\config\\"
    if is_coop == "1":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop\\" + index_write +"\\config\\"
    ###100%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('100%')
    f.close()
    ###100%進度點###

    ###########json return#############
    try:
        index_write = logic_structure['index_write']
    except:
        index_write=0

    if index_write>=1:
        #create fold
        create_date = time.strftime("%Y-%m-%d %H:%M:%S")

        if logic_structure['coop']=="1":
            path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop'
        else:
            path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer'

        if not os.path.isdir(path_orig):
            os.mkdir(path_orig)

        path = path_orig+'/'+str(index_write)
        if not os.path.isdir(path):
            os.mkdir(path)

        Patient_output = Patient_output.to_json(path+"\\Patient (basic information & comorbidity).json",orient='records',date_format = 'iso', double_precision=3)
        Disease_output = Disease_output.to_json(path+"\\Disease (disease diagnosis).json",orient='records',date_format = 'iso', double_precision=3)
        Labs_output = Labs_output.to_json(path+"\\Labs (disease related lab data & biomarkers).json",orient='records',date_format = 'iso', double_precision=3)
        Treatment_1_output = Treatment_1_output.to_json(path+"\\Treatment_1 (first primary treatment plan).json",orient='records',date_format = 'iso', double_precision=3)
        Outpatient_Drug1 = Outpatient_Drug1.to_json(path+"\\Treatment_2 (cancer related medication from outpatients).json",orient='records',date_format = 'iso', double_precision=3)
        Inpatient_Drug1 = Inpatient_Drug1.to_json(path+"\\Treatment_3 (cancer related medication from inpatients).json",orient='records',date_format = 'iso', double_precision=3)
        Treatment_4_output = Treatment_4_output.to_json(path+"\\Treatment_4 (cancer related surgery & radiotherapy).json",orient='records',date_format = 'iso', double_precision=3)
        Outcome_output = Outcome_output.to_json(path+"\\Outcome (treatment outcome).json",orient='records',date_format = 'iso', double_precision=3)

        def getdirsize(dir):
            size = 0
            for root, dirs, files in os.walk(dir):
                size += sum([getsize(join(root, name)) for name in files])
                
            return round(size/1024)

        size = getdirsize(path)

        #DB
        if logic_structure['coop']=="1":
            sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\NBCTDataDb.db'
            conn = sqlite3.connect(sqldb)
            cursor = conn.cursor()

            sql_update_query = """Update CoopJsonData set ModifyTime =""" + "'" + create_date + "'" +','+"""Size ="""+"'"+ str(size) +"'"  + """where [Index] =""" +"'"+str(index_write)+"'"
            cursor.execute(sql_update_query)
            conn.commit()

        else:
            sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\NBCTDataDb.db'
            conn = sqlite3.connect(sqldb)
            cursor = conn.cursor()

            sql_update_query = """Update JsonData set ModifyTime =""" + "'" + create_date + "'" +','+"""Size ="""+"'"+ str(size) +"'"  + """where [Index] =""" +"'"+str(index_write)+"'"
            cursor.execute(sql_update_query)
            conn.commit()

        return("OK")

    else:
        Patient_output = Patient_output.to_json(orient='records',date_format = 'iso', double_precision=3)
        Disease_output = Disease_output.to_json(orient='records',date_format = 'iso', double_precision=3)
        Labs_output = Labs_output.to_json(orient='records',date_format = 'iso', double_precision=3)
        Treatment_1_output = Treatment_1_output.to_json(orient='records',date_format = 'iso', double_precision=3)
        Outpatient_Drug1 = Outpatient_Drug1.to_json(orient='records',date_format = 'iso', double_precision=3)
        Inpatient_Drug1 = Inpatient_Drug1.to_json(orient='records',date_format = 'iso', double_precision=3)
        Treatment_4_output = Treatment_4_output.to_json(orient='records',date_format = 'iso', double_precision=3)
        Outcome_output = Outcome_output.to_json(orient='records',date_format = 'iso', double_precision=3)

        return jsonify(
            Patient_output,
            Disease_output,
            Labs_output,
            Treatment_1_output, #Treatment_1_output
            Outpatient_Drug1, #Treatment_2_output
            Inpatient_Drug1, #Treatment_3_output
            Treatment_4_output, #Treatment_4_output
            Outcome_output
            )

@app.route("/chart", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject14():
    return  ("api refused connection, please try again with 'get'")

@app.route("/json_to_excel", methods=['post'])
def post15():
    global fileDir,left_columns
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    query = query.split(',',1)
    fileDir = query[0]
    left_columns = query[1]
    mix_sult_14 = json_to_excel(fileDir,left_columns)
    return(mix_sult_14)

@app.route("/json_to_excel", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject15():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Version", methods=['post','get'])
def post16():
    mix_sult_16 = Version()
    return jsonify({'Version':mix_sult_16[0],'dbstate':mix_sult_16[1]})

@app.route("/Version", methods=['put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject16():
    return  ("api refused connection, please try again with 'get'")

@app.route("/BiobankID_grouped", methods=['post'])
def post17():
    query = request.get_data()
    query = query.decode(encoding='utf-8', errors='strict')
    mix_sult_17 = BiobankID_grouped(query)
    return mix_sult_17

@app.route("/BiobankID_grouped", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject17():
    return  ("api refused connection, please try again with 'post'")

@app.route("/Coop", methods=['post'])
def post18():
    global logic_structure
    query = request.get_data()
    logic_structure = request.get_json()

    mix_sult_10 = C_CANCER_plus(logic_structure)
    df_CRLF = mix_sult_10[0]
    AE_row_count_before = mix_sult_10[1]
    # AE_row_count_after = mix_sult_10[2]
    BE_row_count_before = mix_sult_10[3]
    # BE_row_count_after = mix_sult_10[4]
    TOTFAO1_match_df = mix_sult_10[5]
    TOTFBO1_matchD_df = mix_sult_10[6]
    TOTFBO1_matchS_df = mix_sult_10[7]
    TOTFBO1_matchC_df = mix_sult_10[8]
    LABM1_Check_match_df = mix_sult_10[9]
    LABM1_Surgery_match_df = mix_sult_10[10]
    df_CASE_match_df = mix_sult_10[11]
    # df_final = mix_sult_10[12]
    # df_CRLF_demo_c = mix_sult_10[13].reset_index()
    df_DEATH = mix_sult_10[14]

    del mix_sult_10
    gc.collect()

    #############################################
    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\drugs.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()
    #############################################
    def date_removezero(x):
        if x == "0" or 0:
            return("")
        else:
            return(x)
    #############################################
    ############ Patient demographic ############
    Patient_demographic = df_CRLF.copy()
    try:
        Patient_demographic = Patient_demographic[['id','dbirth','sex','height','weight','hospid','resid']]
        Patient_demographic.rename(columns = {'id':'hash_id','dbirth':'Year_of_Birth','sex':'Gender','height':'Height_cm','weight':'Weight_kg','hospid':'Hosp_hash','resid':'Residence_code'}, inplace = True)
    except:
        pass
    ############ Cancer Characteristic (Newly diagnosis) ############
    Cancer_Characteristic_Newly_diagnosis = df_CRLF.copy()
    try:
        Cancer_Characteristic_Newly_diagnosis = Cancer_Characteristic_Newly_diagnosis[['id','didiag','site','lateral','hist','behavior','ajcc_ed','cstage','ct','cn','cm','pstage','pt','pn','pm','ostage','ostagec','ostagep']]
        Cancer_Characteristic_Newly_diagnosis.rename(columns = {'id':'hash_id','didiag':'Diagnosis_Date','site':'Primary_site','lateral':'Laterality','hist':'Histology','behavior':'Behavior','ajcc_ed':'AJCC_ed','cstage':'Clinical_stage','ct':'Clinical_t','cn':'Clinical_n','cm':'Clinical_m','pstage':'Pathologic_stage','pt':'Pathologic_t','pn':'Pathologic_n','pm':'Pathologic_m','ostage':'Other_stage_system','ostagec':'Other_clinical_stage','ostagep':'Other_pathologic_stage'}, inplace = True)
    except:
        pass
    ############ Cancer Characteristic (First recurrence) ############
    Cancer_Characteristic_First_recurrence = df_CRLF.copy()
    try:
        Cancer_Characteristic_First_recurrence = Cancer_Characteristic_First_recurrence[['id','site','lateral','hist','behavior']]
        Cancer_Characteristic_First_recurrence.rename(columns = {'id':'hash_id','site':'Primary_site','lateral':'Laterality','hist':'Histology','behavior':'Behavior'}, inplace = True)
    except:
        pass

    ############ Comorbidities ############
    Comorbidities = pd.DataFrame()
    cci=['Myocardial infarction','Congestive heart failure','Peripheral vascular disease','Cerebrovascular disease','Dementia','Chronic pulmonary disease','Rheumatic disease','Peptic ulcer disease','Mild liver disease','Diabetes without chronic complication','Diabetes with chronic complication','Hemiplegia or paraplegia','Renal disease','any malignancy','moderate or severe liver disease','Metastatic solid tumor','AIDS/HIV']
    def Comorbidity_bine(x):
            cc=0
            for cc in range(len(cci)):
                if  x[cci[cc]+"_x"]>=2 or x[cci[cc]+"_y"]>=1:
                    x[cci[cc]+"_binary"] = 1
                    cc+=1
                else:
                    x[cci[cc]+"_binary"] = 0
                    cc+=1
            return(x)
    #0508

    def ABE_fill_function(df):
        ABE_fill = df_CRLF[['id']].copy()
        ABE_fill['Myocardial infarction'] = 0
        ABE_fill['Congestive heart failure'] = 0
        ABE_fill['Peripheral vascular disease'] = 0
        ABE_fill['Cerebrovascular disease'] = 0
        ABE_fill['Dementia'] = 0
        ABE_fill['Chronic pulmonary disease'] = 0
        ABE_fill['Rheumatic disease'] = 0
        ABE_fill['Peptic ulcer disease'] = 0
        ABE_fill['Mild liver disease'] = 0
        ABE_fill['Diabetes without chronic complication'] = 0
        ABE_fill['Diabetes with chronic complication'] = 0
        ABE_fill['Hemiplegia or paraplegia'] = 0
        ABE_fill['Renal disease'] = 0
        ABE_fill['any malignancy'] = 0
        ABE_fill['moderate or severe liver disease'] = 0
        ABE_fill['Metastatic solid tumor'] = 0
        ABE_fill['AIDS/HIV'] = 0

        if 'no_match' in df.columns:
            df = ABE_fill

        return(df)

    try:
        AE_row_count_before = ABE_fill_function(AE_row_count_before)
        BE_row_count_before = ABE_fill_function(BE_row_count_before)
        Comorbidities = pd.merge(AE_row_count_before, BE_row_count_before, how='outer', on=['id'], indicator=False).fillna(np.nan)
        Comorbidities = Comorbidities.apply(Comorbidity_bine, axis=1)

        # 全部id
        fill_allzero = df_CRLF[['id']].copy()
        # 判断是否存在新行
        new_rows = fill_allzero[~fill_allzero['id'].isin(Comorbidities['id'])]
        Comorbidities = pd.concat([Comorbidities, new_rows], ignore_index=True).fillna("0")

        d=0
        for d in range(len(cci)):
            try:
                Comorbidities.drop([cci[d]+"_x", cci[d]+"_y"], axis=1, inplace=True)
            except:
                pass
            d+=1
        
        Comorbidities.rename(columns = {'id':'hash_id'}, inplace = True)
    except:
        pass

    #04/24 RE
    try:
        Comorbidities.rename(columns = {'Myocardial infarction_binary':'com_1','Congestive heart failure_binary':'com_2','Peripheral vascular disease_binary':'com_3','Cerebrovascular disease_binary':'com_4','Dementia_binary':'com_5','Chronic pulmonary disease_binary':'com_6','Rheumatic disease_binary':'com_7','Peptic ulcer disease_binary':'com_8','Mild liver disease_binary':'com_9','Diabetes without chronic complication_binary':'com_10','Diabetes with chronic complication_binary':'com_11','Hemiplegia or paraplegia_binary':'com_12','Renal disease_binary':'com_13','any malignancy_binary':'com_14','moderate or severe liver disease_binary':'com_15','Metastatic solid tumor_binary':'com_16','AIDS/HIV_binary':'com_17'}, inplace = True)
        
    except:
        pass    
    ############ Laboratory results ############
    LAB_lab1 = pd.DataFrame()
    LAB_sug1 = pd.DataFrame()

    LAB_lab = LABM1_Check_match_df.copy()
    LAB_sug = LABM1_Surgery_match_df.copy()
    Laboratory_results = pd.DataFrame()

    try:
        LAB_lab1 = LAB_lab[['id','h18','h23','h25','r2','r3','r4','r5','r6_1','r6_2','r10']]
    except:
        LAB_lab1
    try:
        LAB_sug1 = LAB_sug[['id','h18','h23','h25','r2','r3','r4','r5','r6_1','r6_2','r10']]
    except:
        LAB_sug1

    Laboratory_results = pd.concat([Laboratory_results, LAB_lab1],axis=0)
    Laboratory_results = pd.concat([Laboratory_results, LAB_sug1],axis=0)
    try:
        Laboratory_results.rename(columns = {'id':'hash_id','h18':'Test_code','h23':'Test_date','h25':'Specimen_source','r2':'Lab_name','r3':'Method','r4':'Lab value','r5':'Unit_of_measure','r6_1':'Low_normal_range','r6_2':'High_normal_range','r10':'Result_date'}, inplace = True)
        Laboratory_results['Test_date'] = Laboratory_results['Test_date'].astype(str).str.replace('-','').apply(date_removezero)
        Laboratory_results['Result_date'] = Laboratory_results['Result_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Biomarker test ############ 
    LAB_lab1 = pd.DataFrame()
    LAB_sug1 = pd.DataFrame()

    LAB_lab = LABM1_Check_match_df.copy()
    LAB_sug = LABM1_Surgery_match_df.copy()
    Biomarker_test = pd.DataFrame()

    try:
        LAB_lab1 = LAB_lab[['id','h18','h23','h25','r2','r3','r10','r4']]
    except:
        LAB_lab1
    try:
        LAB_sug1 = LAB_sug[['id','h18','h23','h25','r2','r3','r10','r4']]
    except:
        LAB_sug1

    Biomarker_test = pd.concat([Biomarker_test, LAB_lab1],axis=0)
    Biomarker_test = pd.concat([Biomarker_test, LAB_sug1],axis=0)
    try:
        Biomarker_test.rename(columns = {'id':'hash_id','h18':'Test_code','h23':'Test_date','h25':'Specimen_type_site','r2':'Test_name','r3':'Test_method','r10':'Result_date','r4':'Data_value'}, inplace = True)
        Biomarker_test['Test_date'] = Biomarker_test['Test_date'].astype(str).str.replace('-','').apply(date_removezero)
        Biomarker_test['Result_date'] = Biomarker_test['Result_date'].astype(str).str.replace('-','').apply(date_removezero)       
    except:
        pass

    ############ Cancer related medication (inpatient) ############
    Cancer_related_medication_inpatient = TOTFBO1_matchD_df.copy()
    try:
        Cancer_related_medication_inpatient = TOTFBO1_matchD_df[['id','p3','d10','d11','p5','p6','p7']].copy()
        Cancer_related_medication_inpatient['Dosage_unit'] = ''
    except:
        pass

    try:
        Cancer_related_medication_inpatient = Cancer_related_medication_inpatient.loc[Cancer_related_medication_inpatient['p3'].str.len()==10]
        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p3'}, inplace = True)
        Cancer_related_medication_inpatient = pd.merge(Cancer_related_medication_inpatient, ATC, how='left', on=['p3'], indicator=False)
        Cancer_related_medication_inpatient.drop('p3',axis=1,inplace=True)
        Cancer_related_medication_inpatient.rename(columns = {'id':'hash_id','ATC_CODE':'Medication','d10':'Rx_start_date','d11':'Rx_end_date','p5':'Dosage','p6':'Frequency','p7':'Route'}, inplace = True)
        Cancer_related_medication_inpatient = Cancer_related_medication_inpatient[['hash_id','Medication','Rx_start_date','Rx_end_date','Dosage','Dosage_unit','Frequency','Route']].copy()
        Cancer_related_medication_inpatient['Rx_start_date'] = Cancer_related_medication_inpatient['Rx_start_date'].astype(str).str.replace('-','').apply(date_removezero)
        Cancer_related_medication_inpatient['Rx_end_date'] = Cancer_related_medication_inpatient['Rx_end_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Cancer related medication (outpatient+ER) ############
    Cancer_related_medication_outpatientER = TOTFAO1_match_df.copy()
    try:
        Cancer_related_medication_outpatientER = TOTFAO1_match_df[['id','p4','d9','p1','p5','p7','p9']].copy()
        Cancer_related_medication_outpatientER['Dosage_unit'] = ''
    except:
        pass
    
    try:
        Cancer_related_medication_outpatientER = Cancer_related_medication_outpatientER.loc[Cancer_related_medication_outpatientER['p4'].str.len()==10]
        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p4'}, inplace = True)
        Cancer_related_medication_outpatientER = pd.merge(Cancer_related_medication_outpatientER, ATC, how='left', on=['p4'], indicator=False)
        Cancer_related_medication_outpatientER.drop('p4',axis=1,inplace=True)
        Cancer_related_medication_outpatientER.rename(columns = {'id':'hash_id','ATC_CODE':'Medication','d9':'Rx_start_date','p1':'Drug_day','p5':'Dosage','p7':'Frequency','p9':'Route'}, inplace = True)
        Cancer_related_medication_outpatientER = Cancer_related_medication_outpatientER[['hash_id','Medication','Rx_start_date','Drug_day','Dosage','Dosage_unit','Frequency','Route']].copy()
        Cancer_related_medication_outpatientER['Rx_start_date'] = Cancer_related_medication_outpatientER['Rx_start_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Comedication (inpatient) ############
    Comedication_inpatient = TOTFBO1_matchD_df.copy()
    try:
        Comedication_inpatient = TOTFBO1_matchD_df[['id','p3','d10','d11','p5','p6','p7']].copy()
        Comedication_inpatient['Dosage_unit'] = ''
    except:
        pass

    try:
        Comedication_inpatient = Comedication_inpatient.loc[Comedication_inpatient['p3'].str.len()==10]
        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p3'}, inplace = True)
        Comedication_inpatient = pd.merge(Comedication_inpatient, ATC, how='left', on=['p3'], indicator=False)
        Comedication_inpatient.drop('p3',axis=1,inplace=True)
        Comedication_inpatient.rename(columns = {'id':'hash_id','ATC_CODE':'Medication','d10':'Rx_start_date','d11':'Rx_end_date','p5':'Dosage','p6':'Frequency','p7':'Route'}, inplace = True)
        Comedication_inpatient = Comedication_inpatient[['hash_id','Medication','Rx_start_date','Rx_end_date','Dosage','Dosage_unit','Frequency','Route']].copy()
        Comedication_inpatient['Rx_start_date'] = Comedication_inpatient['Rx_start_date'].astype(str).str.replace('-','').apply(date_removezero)
        Comedication_inpatient['Rx_end_date'] = Comedication_inpatient['Rx_end_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Comedication (outpatient+ER) ############
    Comedication_outpatient_ER = TOTFAO1_match_df.copy()
    try:
        Comedication_outpatient_ER = TOTFAO1_match_df[['id','p4','d9','p1','p5','p7','p9']].copy()
        Comedication_outpatient_ER['Dosage_unit'] = ''
    except:
        pass

    try:
        Comedication_outpatient_ER = Comedication_outpatient_ER.loc[Comedication_outpatient_ER['p4'].str.len()==10]
        ATC = pd.read_sql("SELECT ATC_CODE, Drug_code FROM new_drugs", conn)
        ATC.rename(columns = {'Drug_code':'p4'}, inplace = True)
        Comedication_outpatient_ER = pd.merge(Comedication_outpatient_ER, ATC, how='left', on=['p4'], indicator=False)
        Comedication_outpatient_ER.drop('p4',axis=1,inplace=True)
        Comedication_outpatient_ER.rename(columns = {'id':'hash_id','ATC_CODE':'Medication','d9':'Rx_start_date','p1':'Drug_day','p5':'Dosage','p7':'Frequency','p9':'Route'}, inplace = True)
        Comedication_outpatient_ER = Comedication_outpatient_ER[['hash_id','Medication','Rx_start_date','Drug_day','Dosage','Dosage_unit','Frequency','Route']].copy()
        Comedication_outpatient_ER['Rx_start_date'] = Comedication_outpatient_ER['Rx_start_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Cancer related radiotherapy ############
    Cancer_related_radiotherapy = df_CRLF.copy()
    try:
        Cancer_related_radiotherapy = Cancer_related_radiotherapy[['id','drt_1st','drt_end']].copy()
        Cancer_related_radiotherapy.rename(columns = {'id':'hash_id','drt_1st':'RT_start_date','drt_end':'RT_end_date'}, inplace = True)
        Cancer_related_radiotherapy['Rx_start_date'] = Cancer_related_radiotherapy['Rx_start_date'].astype(str).str.replace('-','').apply(date_removezero)
    except:
        pass

    ############ Cancer related surgery ############
    Cancer_related_surgery = df_CRLF.copy()
    try:
        Cancer_related_surgery = Cancer_related_surgery[['id','optype_h','dop_mds']].copy()
        Cancer_related_surgery.rename(columns = {'id':'hash_id','optype_h':'Surgical_procedure1','dop_mds':'Surgery_date1'}, inplace = True)
        Cancer_related_surgery['Surgery_date1'] = Cancer_related_surgery['Surgery_date1'].astype(str).str.replace('-','').replace('nan','').apply(date_removezero)
    except:
        pass

    ############ Cancer related surgery2 ############
    Cancer_related_surgery2 = pd.DataFrame()
    Cancer_related_surgery_AE = TOTFAO1_match_df.copy()
    try:
        Cancer_related_surgery_AE = Cancer_related_surgery_AE[['id','p4','d9']].copy()
        Cancer_related_surgery_AE = Cancer_related_surgery_AE.loc[Cancer_related_surgery_AE['p4'].str.len()<10]
        Cancer_related_surgery_AE = Cancer_related_surgery_AE.loc[Cancer_related_surgery_AE['p4'].str.len()>=6]
        Cancer_related_surgery_AE.rename(columns = {'id':'hash_id','p4':'Surgical_procedure2','d9':'Surgery_date2'}, inplace = True)
        Cancer_related_surgery_AE['Surgery_date2'] = Cancer_related_surgery_AE['Surgery_date2'].astype(str).str.replace('-','').replace('nan','').apply(date_removezero)
    except:
        pass

    Cancer_related_surgery_BE = TOTFBO1_matchS_df.copy()
    try:
        Cancer_related_surgery_BE = Cancer_related_surgery_BE[['id','p3','d10']].copy()
        Cancer_related_surgery_BE = Cancer_related_surgery_BE.loc[Cancer_related_surgery_BE['p3'].str.len()<10]
        Cancer_related_surgery_BE = Cancer_related_surgery_BE.loc[Cancer_related_surgery_BE['p3'].str.len()>=6]
        Cancer_related_surgery_BE.rename(columns = {'id':'hash_id','p3':'Surgical_procedure2','d10':'Surgery_date2'}, inplace = True)
        Cancer_related_surgery_BE['Surgery_date2'] = Cancer_related_surgery_BE['Surgery_date2'].astype(str).str.replace('-','').replace('nan','').apply(date_removezero)
        Cancer_related_surgery2 = pd.concat([Cancer_related_surgery_AE,Cancer_related_surgery_BE],axis=0)
    except:
        pass

    ############ Death data ############
    Death_data = df_DEATH.copy()
    try:
        Death_data = Death_data[['id','d4']].copy()
        Death_data.rename(columns = {'id':'hash_id','d4':'Date_of_death'}, inplace = True)
        Death_data['Date_of_death'] = Death_data['Date_of_death'].astype(str).str.replace('-','').replace('nan','').apply(date_removezero)
    except:
        pass

    conn.close()
    #################################### 0504 orig id query data
    search_id = logic_structure['search_id']

    sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\BiobankData.db'
    conn = sqlite3.connect(sqldb)  
    cursor = conn.cursor()
    
    # orig_CRLF = pd.read_sql("SELECT * FROM " + "CRLF", conn)
    # orig_CRLF.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_CRLF = orig_CRLF.query(search_id,engine='python')

    # orig_CASE = pd.read_sql("SELECT * FROM " + "[CASE]", conn)
    # orig_CASE.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_CASE = orig_CASE.query(search_id,engine='python')

    # orig_DEATH = pd.read_sql("SELECT * FROM " + "DEATH", conn)
    # orig_DEATH.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_DEATH = orig_DEATH.query(search_id,engine='python')

    # orig_LABM1 = pd.read_sql("SELECT * FROM " + "LABM1", conn)
    # orig_LABM1.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_LABM1.rename(columns = {'h9':'id'}, inplace = True)
    # orig_LABM1 = orig_LABM1.query(search_id,engine='python')

    orig_TOTFAE = pd.read_sql("SELECT * FROM " + "TOTFAE", conn)
    orig_TOTFAE.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    orig_TOTFAE.rename(columns = {'d3':'id'}, inplace = True)
    orig_TOTFAE = orig_TOTFAE.query(search_id,engine='python')

    # orig_TOTFAO1 = pd.read_sql("SELECT * FROM " + "TOTFAO1", conn)
    # orig_TOTFAO1.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_TOTFAO1.rename(columns = {'d3':'id'}, inplace = True)
    # orig_TOTFAO1 = orig_TOTFAO1.query(search_id,engine='python')

    orig_TOTFBE = pd.read_sql("SELECT * FROM " + "TOTFBE", conn)
    orig_TOTFBE.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    orig_TOTFBE.rename(columns = {'d3':'id'}, inplace = True)
    orig_TOTFBE = orig_TOTFBE.query(search_id,engine='python')

    # orig_TOTFBO1 = pd.read_sql("SELECT * FROM " + "TOTFBO1", conn)
    # orig_TOTFBO1.drop(['Index','verify','IsUploadHash','CreateTime','ModifyTime'],axis=1,inplace=True)
    # orig_TOTFBO1.rename(columns = {'d3':'id'}, inplace = True)
    # orig_TOTFBO1 = orig_TOTFBO1.query(search_id,engine='python')

    ####################################
    ########### Null table fill ########
    def Null_table(df):
        if df.empty == True:
            df = pd.DataFrame({'no_match':['not found cohort']})
            return(df)
        else:
            df = df.drop_duplicates()
            return(df)
    Patient_demographic = Null_table(Patient_demographic)
    Cancer_Characteristic_Newly_diagnosis = Null_table(Cancer_Characteristic_Newly_diagnosis)
    Cancer_Characteristic_First_recurrence = Null_table(Cancer_Characteristic_First_recurrence)
    Comorbidities = Null_table(Comorbidities)
    Laboratory_results = Null_table(Laboratory_results)
    Biomarker_test = Null_table(Biomarker_test)
    Cancer_related_medication_inpatient = Null_table(Cancer_related_medication_inpatient)
    Cancer_related_medication_outpatientER = Null_table(Cancer_related_medication_outpatientER)
    Comedication_inpatient = Null_table(Comedication_inpatient)
    Comedication_outpatient_ER = Null_table(Comedication_outpatient_ER)
    Cancer_related_radiotherapy = Null_table(Cancer_related_radiotherapy)
    Cancer_related_surgery = Null_table(Cancer_related_surgery)
    Cancer_related_surgery2 = Null_table(Cancer_related_surgery2)
    Death_data = Null_table(Death_data)

    # orig_CRLF = Null_table(orig_CRLF)
    # orig_CASE = Null_table(orig_CASE)
    # orig_DEATH = Null_table(orig_DEATH)
    # orig_LABM1 = Null_table(orig_LABM1)
    orig_TOTFAE = Null_table(orig_TOTFAE)
    # orig_TOTFAO1 = Null_table(orig_TOTFAO1)
    orig_TOTFBE = Null_table(orig_TOTFBE)
    # orig_TOTFBO1 = Null_table(orig_TOTFBO1)
    ####################################
    #write index path
    index_write = str(logic_structure['index_write'])
    is_coop = str(logic_structure['coop'])
    if is_coop == "0":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer\\" + index_write +"\\config\\"
    if is_coop == "1":
        plug_path = "C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop\\" + index_write +"\\config\\"
    ###100%進度點###
    process_path = plug_path+'C_process.txt'
    f = open(process_path, 'w')
    f.write('100%')
    f.close()
    ###100%進度點###

    ###########json return#############
    try:
        index_write = logic_structure['index_write']
    except:
        index_write=0

    if index_write>=1:
        #create fold
        create_date = time.strftime("%Y-%m-%d %H:%M:%S")

        if logic_structure['coop']=="1":
            path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_coop'
        else:
            path_orig = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\save_json_cancer'

        if not os.path.isdir(path_orig):
            os.mkdir(path_orig)

        path = path_orig+'/'+str(index_write)
        if not os.path.isdir(path):
            os.mkdir(path)

        Patient_demographic = Patient_demographic.to_json(path+"\\Patient demographic.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_Characteristic_Newly_diagnosis = Cancer_Characteristic_Newly_diagnosis.to_json(path+"\\Cancer Characteristic (Newly diagnosis).json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_Characteristic_First_recurrence = Cancer_Characteristic_First_recurrence.to_json(path+"\\Cancer Characteristic (First recurrence).json",orient='records',date_format = 'iso', double_precision=3)
        Comorbidities = Comorbidities.to_json(path+"\\Comorbidities.json",orient='records',date_format = 'iso', double_precision=3)
        Laboratory_results = Laboratory_results.to_json(path+"\\Laboratory results.json",orient='records',date_format = 'iso', double_precision=3)
        Biomarker_test = Biomarker_test.to_json(path+"\\Biomarker test.json",orient='records',date_format = 'None', double_precision=3)
        Cancer_related_medication_inpatient = Cancer_related_medication_inpatient.to_json(path+"\\Cancer related medication inpatient.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_medication_outpatientER = Cancer_related_medication_outpatientER.to_json(path+"\\Cancer related medication outpatientER.json",orient='records',date_format = 'iso', double_precision=3)
        Comedication_inpatient = Comedication_inpatient.to_json(path+"\\Comedication inpatient.json",orient='records',date_format = 'iso', double_precision=3)
        Comedication_outpatient_ER = Comedication_outpatient_ER.to_json(path+"\\Comedication outpatient ER.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_radiotherapy = Cancer_related_radiotherapy.to_json(path+"\\Cancer related radiotherapy.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_surgery = Cancer_related_surgery.to_json(path+"\\Cancer related surgery.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_surgery2 = Cancer_related_surgery2.to_json(path+"\\Cancer related surgery2.json",orient='records',date_format = 'iso', double_precision=3)
        Death_data = Death_data.to_json(path+"\\Death data.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_CRLF = orig_CRLF.to_json(path+"\\orig_CRLF.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_CASE = orig_CASE.to_json(path+"\\orig_CASE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_DEATH = orig_DEATH.to_json(path+"\\orig_DEATH.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_LABM1 = orig_LABM1.to_json(path+"\\orig_LABM1.json",orient='records',date_format = 'iso', double_precision=3)
        orig_TOTFAE = orig_TOTFAE.to_json(path+"\\orig_TOTFAE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_TOTFAO1 = orig_TOTFAO1.to_json(path+"\\orig_TOTFAO1.json",orient='records',date_format = 'iso', double_precision=3)
        orig_TOTFBE = orig_TOTFBE.to_json(path+"\\orig_TOTFBE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_TOTFBO1 = orig_TOTFBO1.to_json(path+"\\orig_TOTFBO1.json",orient='records',date_format = 'iso', double_precision=3)

        def getdirsize(dir):
            size = 0
            for root, dirs, files in os.walk(dir):
                size += sum([getsize(join(root, name)) for name in files])
                
            return round(size/1024)

        size = getdirsize(path)

        #DB
        if logic_structure['coop']=="1":
            sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\NBCTDataDb.db'
            conn = sqlite3.connect(sqldb)
            cursor = conn.cursor()

            sql_update_query = """Update CoopJsonData set ModifyTime =""" + "'" + create_date + "'" +','+"""Size ="""+"'"+ str(size) +"'"  + """where [Index] =""" +"'"+str(index_write)+"'"
            cursor.execute(sql_update_query)
            conn.commit()

        else:
            sqldb = 'C:\\inetpub\\wwwroot\\Hospital_Converter\\App_Data\\NBCTDataDb.db'
            conn = sqlite3.connect(sqldb)
            cursor = conn.cursor()

            sql_update_query = """Update JsonData set ModifyTime =""" + "'" + create_date + "'" +','+"""Size ="""+"'"+ str(size) +"'"  + """where [Index] =""" +"'"+str(index_write)+"'"

            cursor.execute(sql_update_query)
            conn.commit()

        conn.close()
        return("OK")

    else:
        Patient_demographic = Patient_demographic.to_json(path+"\\Patient demographic.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_Characteristic_Newly_diagnosis = Cancer_Characteristic_Newly_diagnosis.to_json(path+"\\Cancer Characteristic (Newly diagnosis).json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_Characteristic_First_recurrence = Cancer_Characteristic_First_recurrence.to_json(path+"\\Cancer Characteristic (First recurrence).json",orient='records',date_format = 'iso', double_precision=3)
        Comorbidities = Comorbidities.to_json(path+"\\Comorbidities.json",orient='records',date_format = 'iso', double_precision=3)
        Laboratory_results = Laboratory_results.to_json(path+"\\Laboratory results.json",orient='records',date_format = 'iso', double_precision=3)
        Biomarker_test = Biomarker_test.to_json(path+"\\Biomarker test.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_medication_inpatient = Cancer_related_medication_inpatient.to_json(path+"\\Cancer related medication inpatient.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_medication_outpatientER = Cancer_related_medication_outpatientER.to_json(path+"\\Cancer related medication outpatientER.json",orient='records',date_format = 'iso', double_precision=3)
        Comedication_inpatient = Comedication_inpatient.to_json(path+"\\Comedication inpatient.json",orient='records',date_format = 'iso', double_precision=3)
        Comedication_outpatient_ER = Comedication_outpatient_ER.to_json(path+"\\Comedication outpatient ER.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_radiotherapy = Cancer_related_radiotherapy.to_json(path+"\\Cancer related radiotherapy.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_surgery = Cancer_related_surgery.to_json(path+"\\Cancer related surgery.json",orient='records',date_format = 'iso', double_precision=3)
        Cancer_related_surgery2 = Cancer_related_surgery2.to_json(path+"\\Cancer_related_surgery2.json",orient='records',date_format = 'iso', double_precision=3)
        Death_data = Death_data.to_json(path+"\\Death_data.json",orient='records',date_format = 'iso', double_precision=3)
        Death_data = Death_data.to_json(path+"\\Death data.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_CRLF = orig_CRLF.to_json(path+"\\orig_CRLF.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_CASE = orig_CASE.to_json(path+"\\orig_CASE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_DEATH = orig_DEATH.to_json(path+"\\orig_DEATH.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_LABM1 = orig_LABM1.to_json(path+"\\orig_LABM1.json",orient='records',date_format = 'iso', double_precision=3)
        orig_TOTFAE = orig_TOTFAE.to_json(path+"\\orig_TOTFAE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_TOTFAO1 = orig_TOTFAO1.to_json(path+"\\orig_TOTFAO1.json",orient='records',date_format = 'iso', double_precision=3)
        orig_TOTFBE = orig_TOTFBE.to_json(path+"\\orig_TOTFBE.json",orient='records',date_format = 'iso', double_precision=3)
        # orig_TOTFBO1 = orig_TOTFBO1.to_json(path+"\\orig_TOTFBO1.json",orient='records',date_format = 'iso', double_precision=3)

        conn.close()
        return jsonify(
            Patient_demographic,
            Cancer_Characteristic_Newly_diagnosis,
            Cancer_Characteristic_First_recurrence,
            Comorbidities,
            Laboratory_results,
            Biomarker_test,
            Cancer_related_medication_inpatient, 
            Cancer_related_medication_outpatientER,
            Comedication_inpatient,
            Comedication_outpatient_ER,
            Cancer_related_radiotherapy,
            Cancer_related_surgery,
            Cancer_related_surgery2,
            Death_data,
            # orig_CRLF,
            # orig_CASE,
            # orig_DEATH,
            # orig_LABM1,
            orig_TOTFAE,
            # orig_TOTFAO1,
            orig_TOTFBE
            # orig_TOTFBO1
            )

@app.route("/Coop", methods=['get','put','patch','delete','copy','head','options','link','unlink','purge','lock','unlock','propfind','view'])
def reject18():
    return  ("api refused connection, please try again with 'post'")

if __name__=="__main__":
    app.run(host='0.0.0.0', debug = False, port=5000, threaded=True) #processes=True threaded=True