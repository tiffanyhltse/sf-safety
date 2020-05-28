#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:00:53 2020

@author: Jayleen
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import psycopg2
import matplotlib.pyplot as plt

import sklearn as sk
from shapely.geometry import Point, Polygon
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans
# import plotly.express as px

import tensorflow as tf
# import imblearn # python module to balance data set using random over/undersampling

#%%
# Sprint2

def load_pandas(base_dir = "C:/Users/user/Desktop/USF 2020-1/BSDS200/FINAL PROJECT"):
    TEMPORARY_DIR = base_dir + "/TmpDir/"
    file_names = ["Police_Department_Incident_Reports__Historical_2003_to_May_2018",
                  "Police_Department_Incident_Reports__2018_to_Present",
                  "SFO_Weather_2003_to_Present",
                  "Housing_Inventories_2005_to_2018"]
    DF_tdf = []
    
    DF_read = [pd.read_csv(f"{file_names[i]}.csv", sep=',') for i in range(len(file_names))]
    DF_read[0] = DF_read[0].iloc[:, :17]
    DF_read[1] = DF_read[1].iloc[:, :29]
    # DF_read[2] = DF_read[2].iloc[2:, :]
    DF_read[3] = DF_read[3].iloc[:, :18]
    DF_tdf = [DF_read[i].to_csv(TEMPORARY_DIR + f"{file_names[i]}.tdf", sep='\t', index=False) for i in range(len(file_names))]

    return DF_tdf


def load_sql(base_dir = "C:/Users/user/Desktop/USF 2020-1/BSDS200/FINAL PROJECT"):
    TEMPORARY_DIR = base_dir + "/TmpDir/"
    print("Temp directory: ", TEMPORARY_DIR)
    
    file_names = ["Police_Department_Incident_Reports__Historical_2003_to_May_2018",
                  "Police_Department_Incident_Reports__2018_to_Present",
                  "SFO_Weather_2003_to_Present",
                  "Housing_Inventories_2005_to_2018"]

    Datas = [pd.read_csv(f"{file_names[i]}.csv", sep = ',') for i in range(len(file_names))]
    Datas[0] = Datas[0].iloc[:, :17]
    Datas[1] = Datas[1].iloc[:, :29]
    # Datas[2] = Datas[2].iloc[2:, :]
    Datas[3] = Datas[3].iloc[:, :18]
    Datas = [Datas[i].to_csv(TEMPORARY_DIR + f"{file_names[i]}.tdf", sep = '\t', index = False) for i in range(len(file_names))]
    
    
    ### Set up your SQL conection:
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCursor = SQLConn.cursor()
    schema_name = 'sf_safety'
    table_name = ['reports2003_to_2018', 'reports2018_to_present', 'weather_2003_to_present', 'income_2005_to_2018']
    
    for i in range(len(table_name)):
        try:
            SQLCursor.execute("""DROP TABLE %s.%s;""" % (schema_name, table_name[i]))
            SQLConn.commit()
        except psycopg2.ProgrammingError:
            print("!!!CAUTION: Tablenames not found: %s.%s!!!" % (schema_name, table_name[i]))
            SQLConn.rollback()
    
    ## Load into DB
    SQLCursor = SQLConn.cursor()
    table_execute = ["""(IncidntNum int
                      , Category varchar(30)
                      , Descript varchar(80)
                      , DayOfWeek varchar(9)
                      , Date date
                      , Time time
                      , PdDistrict varchar(10)                      
                      , Resolution varchar(60)
                      , Address varchar (60)
                      , X float
                      , Y float
                      , Location varchar(50)
                      , Pdld float
                      , SFFindNbhd float
                      , CurrentPd float
                      , CurrentSd float
                      , Nbhd float
                      )""",
                     
                     """(IncidentDt timestamp
                      , IncidntDate date
                      , IncidntTime time
                      , IncidntYr int
                      , IncidntDOW varchar(9)
                      , ReportDt timestamp
                      , RowID float
                      , IncidntID int
                      , IncidntNum int
                      , CADNum float
                      , ReportTypeCode varchar(3)
                      , ReportTypeDes varchar(20)
                      , FiledOnline varchar(5)
                      , IncidntCode int
                      , IncidntCategory varchar(50)
                      , IncidntSubcategory varchar(80)
                      , IncidntDes varchar(80)
                      , Resolution varchar(60)
                      , Intersection varchar(100)
                      , CNN float
                      , PdDistrict varchar(15)
                      , Nbhd varchar(30)
                      , SupervisorDist float
                      , Latitude float
                      , Longitude float
                      , point varchar(50)
                      , SFFindNbhd float
                      , CurrentPd float
                      , CurrentSd float                     
                      )""",
                     
                     """(Date date
                      , Time varchar(20)
                      , air_temp float
                      , dew_pt_temp float
                      , rel_humidity float
                      , wind varchar(40)
                      , wind_speed float
                      , wind_gust varchar(40)
                      , air_pressure float
                      , PrecipRate float
                      , Condition varchar(40)
                      )""",
                     
                     """(ACTDT date
                      , YR int
                      , YR_QTR varchar(10)
                      , ACTION varchar(40)
                      , APP_NO varchar(30)
                      , FM int
                      , NUMBER int
                      , ST varchar(40)                      
                      , ST_TYPE varchar(10)
                      , BLOCK varchar(10)
                      , LOT varchar(10)
                      , DESCRIPT character varying(100)
                      , BLOCKLOT character varying(3000)
                      , EXT_USE varchar(20)
                      , PROP_USE varchar(50)
                      , AFF_HSG float 
                      , AFF_TARGET varchar(50)
                      , Nbhd varchar(50)
                      )"""]
                     
    for i in range(len(table_name)):
        SQLCursor.execute(f"""
                              CREATE TABLE {schema_name}.{table_name[i]}
                              {table_execute[i]};
                              """)
        SQLConn.commit()
        SQLCursor.execute(f"""GRANT ALL on sf_safety.{table_name[i]} to students;""")
        SQLConn.commit()
        SQL_STATEMENT = f"""        
        COPY {schema_name}.{table_name[i]} FROM STDIN WITH
                  CSV
                  HEADER
                  DELIMITER
                  AS E'\t';        
                  """    
        SQLCursor.copy_expert(sql=SQL_STATEMENT , file = open( TEMPORARY_DIR + f'{file_names[i]}.tdf', 'r'))
        SQLConn.commit()   
 

def tests(base_dir = "C:/Users/user/Desktop/USF 2020-1/BSDS200/FINAL PROJECT"):
  TEMPORARY_DIR = base_dir + "/TmpDir/"
  CONNECTION_STRING = "dbname = 'bsdsclass' user = 'ymoon2' host = 'bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password = 'Divider6'"
  # CONNECTION_STRING = "dbname = 'bsdsclass' user = 'ammuth' host = 'bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password = 'Driver6'"
  file_names = ["Police_Department_Incident_Reports__Historical_2003_to_2017",
                  "Police_Department_Incident_Reports__2018_to_Present",
                  "SFO_Weather_2003_to_Present",
                  "Housing_Inventories_2005_to_2018"]
  table_name = ['reports2003_to_2017', 'reports2018_to_present', 'weather_2003_to_present', 'income_2005_to_2018']
  DFs = [pd.read_csv(TEMPORARY_DIR + f"{file_names[i]}.tdf", sep = '\t') for i in range(len(file_names))]
  
  SQLConn = psycopg2.connect(CONNECTION_STRING)
  SQLCursor = SQLConn.cursor()
  SQL_rows = []
  DFs_rows = [] 
  
  ### Test #1: Verify the number of rows
  for i in range(len(DFs)):
      SQLCursor.execute(f"""Select count(1) from sf_safety.{table_name[i]};""")
      sql_rows = SQLCursor.fetchall()
      SQL_rows.append(sql_rows[0][0])
      DFs_rows.append(DFs[i].shape[0])
      if DFs_rows[i] != SQL_rows[i]:
          print(f"DF{i+1} rows don't match!!!!{DFs_rows[i]}, {SQL_rows[i]}")
      if DFs_rows[i] == SQL_rows[i]:
          print(f"DF{i+1} rows matches :))) {DFs_rows[i]}, {SQL_rows[i]}")    

  ### Test #2: Value count verification
  sql_colimp = ['PdDistrict', 'incidntcategory', 'station', 'st']
  pd_colimp = ['PdDistrict', 'Incident Category', 'station', 'STREET']
  for i in range(len(DFs)):
       SQLCursor.execute(f"""Select {sql_colimp[i]}, count(1) as ct from sf_safety.{table_name[i]} where {sql_colimp[i]} is not null group by 1 order by 2;""")
       sql_rows = SQLCursor.fetchall()
       sql_rows = pd.DataFrame(sql_rows, columns=[f'{sql_colimp[i]}', 'ct']).sort_values(['ct'], ascending = False).reset_index(drop=True)
       DF_rows = DFs[i][f'{pd_colimp[i]}'].value_counts().to_frame().reset_index().rename(columns={f'{pd_colimp[i]}' : 'ct', 'index' : f'{pd_colimp[i]}'}).reset_index(drop=True)
       if(DF_rows.equals(sql_rows)):
           print(f"DF{i+1} rows equal sql rows")
 

def processing_pd(df):
    df.loc[(df.Category == 'WEAPON LAWS') | (df.Category == 'WEAPONS CARRYING ETC') | (df.Category == "WEAPONS OFFENCE"), ['Category']] = 'WEAPONS OFFENSE'
    df.loc[(df.Category == 'SEX OFFENSES, NON FORCIBLE'), ['Category']] = 'SEX OFFENSE'
    df.loc[(df.Category == 'SEX OFFENSES, FORCIBLE'), ['Category']] = 'RAPE'
    df.loc[(df.Category == 'OTHER MISCELLANEOUS') | (df.Category == 'OTHER'), ['Category']] = 'OTHER OFFENSES'

    df.loc[(df.Category == 'OFFENCES AGAINST THE FAMILY AND CHILDREN') | (df.Category == 'FAMILY OFFENSE'), ['Category']] = 'FAMILY OFFENSES'
    df.loc[(df.Category == 'FORGERY AND COUNTERFEITING') | (df.Category == 'BAD CHECKS') | (df.Category == 'EMBEZZLEMENT') | (df.Category == 'FRAUD'), ['Category']] = 'FORGERY/COUNTERFEITING'
    df.loc[(df.Category == 'LARCENY THEFT'), ['Category']] = 'LARCENY/THEFT'
    df.loc[(df.Category == 'SUSPICIOUS'), ['Category']] = 'SUSPICIOUS OCC'
    df.loc[(df.Category == 'WARRANT'), ['Category']] = 'WARRANTS'
    df.loc[(df.Category == 'MOTOR VEHICLE THEFT') | (df.Category == 'MOTOR VEHICLE THEFT?') | (df.Category == 'VEHICLE MISPLACED'), ['Category']] = 'VEHICLE THEFT'
    df.loc[(df.Category =='HUMAN TRAFFICKING (A), COMMERCIAL SEX ACTS') | (df.Category =='HUMAN TRAFFICKING (B), INVOLUNTARY SERVITUDE') 
            | (df.Category =='HUMAN TRAFFICKING, COMMERCIAL SEX ACTS'), ['Category']] = 'HUMAN TRAFFICKING'
    df.loc[(df.Category == 'DRUG VIOLATION') | (df.Category == 'DRUG OFFENSE'), ['Category']] = 'DRUG/NARCOTIC'
    df.loc[(df.Category == 'TRAFFIC VIOLATION ARREST') | (df.Category == 'DRIVING UNDER THE INFLUENCE') | (df.Category == 'TRAFFIC COLLISION'), ['Category']] = 'TRAFFIC'
    df.loc[(df.Category == 'DISORDERLY CONDUCT'), ['Category']] = 'PORNOGRAPHY/OBSCENE MAT'
    df.loc[(df.Category == 'ARSON'), ['Category']] = 'VANDALISM'
    df.loc[(df.Category == 'SEX OFFENSE') | (df.Category == 'RAPE') | (df.Category == 'HOMICIDE'), ['Category']] = 'VIOLENT'
    
    df = df.drop_duplicates(keep = False)
    df = df.drop(df[df.Category.isna() == True].index)
    # these crime categories do not pertain to criminal activities
    df = df.drop(df[(df.Category == 'CASE CLOSURE') | (df.Category == 'NON-CRIMINAL') | (df.Category == 'RECOVERED VEHICLE') | (df.Category == 'SECONDARY CODES')].index)  
    df = df.reset_index().drop('index', axis = 1)
    
    return df
#%%
# Sprint3

#loading the tables by Sql    
CONNECTION_STRING = "dbname = 'bsdsclass' user = 'ymoon2' host = 'bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password = 'Divider6'"
SQLConn = psycopg2.connect(CONNECTION_STRING)
SQLCursor = SQLConn.cursor()

columns = [["IncidntNum", "Category", "DayOfWeek", "Year", "Date", "Time", "Descript", "PdDistrict", "X", "Y", "Resolution"]
           , ['Date', "Time", "AirTemp", "DewTemp", "Humidity", "WindSpeed"]]

executes = ["""
            (SELECT IncidntNum, Category, DayOfWeek, 
                         date_part('year', date) as Year, Date, Time, Descript, PdDistrict, X, Y, Resolution
                   FROM sf_safety.reports2003_to_2018) 
            UNION ALL
            (SELECT IncidntNum, upper(incidntcategory) as Category, incidntdow as DayofWeek, 
                    incidntyr as Year, incidntdate as Date, 
                    incidnttime as Time, upper(incidntdes) as Descript, upper(PdDistrict), 
                    latitude as X, longitude as Y, upper(resolution)    
            FROM sf_safety.reports2018_to_present);
            """,
            """
                  SELECT Date, Time, air_temp as AirTemp, dew_pt_temp as DewTemp, 
                         rel_humidity as Humidity, wind_speed as WindSpeed
                  FROM sf_safety.weather_2003_to_present;
            """] #sf_safety.reports2003_to_2017


SQLCursor.execute(executes[0])
reports2003_present = pd.DataFrame(SQLCursor.fetchall(), columns = columns[0])
SQLCursor.execute(executes[1])
weather2003 = pd.DataFrame(SQLCursor.fetchall(), columns = columns[1])

#%%

reports2003_present = processing_pd(reports2003_present)
reports2003_present['Hour'] = pd.Series(x.hour for x in reports2003_present.Time)
reports2003_present.drop(reports2003_present[reports2003_present['PdDistrict'] == 'OUT OF SF'].index, inplace = True)
weather2003['Time'] = pd.to_datetime(weather2003.Time)
weather2003['Hour'] = pd.Series(x.hour for x in weather2003.Time)

#%%
#### QUESTION 1
#grabbing only the columns we care about for this Q..
reports2003_present_Q1 = reports2003_present.loc[:, ['Category', 'Year', 'PdDistrict']]

#plots with merged dataset
def GEN_top10_crimes(data):
  #GRAB THE TOP 10 CRIMES
  merged_sub1 = data.groupby(['Category']).size().reset_index().rename(columns = {0:'freq'}).nlargest(10, ['freq'])
  top_ten_crimes = merged_sub1.Category
  merged_sub = data

  #SELECT THE ROWS THAT ONLY HAVE THE TOP 10 CATEGORIES FROM OVERALL DF
  sub = data.loc[(data.Year != 2020), ['Category', 'Year']]
  sub = sub.groupby(['Category', 'Year']).size().reset_index().rename(columns = {0 : 'freq'})

  #NOW FOR EACH OF THESE CRIMES, GRAPH FREQUENCY OVER TIME ON LINE GRAPH
  for i in range(0,top_ten_crimes.size):
    l = top_ten_crimes.iloc[i]
    p = sub.loc[(sub.Category == l), :]
    plt.plot(p.Year, p.freq, label = l)
  plt.legend(loc='upper center', bbox_to_anchor=(1.28, 0.8), shadow=True, ncol=1)
  plt.xlabel('Year')
  plt.ylabel('Crime Count')
  plt.title('Top 10 Crimes (2003-present)', fontweight = 'bold', fontsize = 12)
  plt.show()
  plt.savefig('Top 10 Crimes (2003-present).png')

GEN_top10_crimes(reports2003_present_Q1)


#%%
### QUESTION 2

reports2003_present_Q2 = reports2003_present.loc[:, ['IncidntNum', 'Date', 'Hour', 'DayOfWeek', 'PdDistrict']]
weather2003_Q2 = weather2003.loc[:,['Date', 'Hour', 'AirTemp']]
reports_weather = pd.merge(reports2003_present_Q2, weather2003_Q2, how = 'left', on = ['Date', 'Hour'])


def GEN_weather_crimes(reports_weather):
  # drop missing weather data  
  allTemps = reports_weather.loc[(reports_weather.AirTemp.isna() == False), :].reset_index()
  allTemps = allTemps.drop('index', axis =1)
  # temp = allTemps.AirTemp
  allTemps.reset_index(level = 0, inplace = True)

  ### this is for stacked histogram according to weekend/weekday
  # dayClass = allTemps.assign(DayOfWeek = 0)
  allTemps.loc[(allTemps.DayOfWeek == 'Monday')|(allTemps.DayOfWeek == 'Tuesday')|(allTemps.DayOfWeek == 'Wednesday')|(allTemps.DayOfWeek == 'Thursday')|(allTemps.DayOfWeek == 'Friday'), 'dayClass'] = 'Weekday'
  allTemps.loc[(allTemps.DayOfWeek == 'Saturday')|(allTemps.DayOfWeek == 'Sunday'), 'dayClass'] = 'Weekend'
  allTemps[['dayClass', 'index', 'IncidntNum']]

  plt.figure(figsize = (9,11))
  allTemps.pivot(columns = 'dayClass').AirTemp.plot(kind = 'hist', stacked = True, edgecolor = 'black', linewidth = 1.2)
  plt.xlabel('Temperature (Celsius)')
  plt.ylabel('Crime Frequency')
  plt.legend(loc='best')
  plt.title('Reported Incidents by Weather Temperature', fontweight = 'bold', fontsize = 12)
  plt.savefig('Reported Incidents by Weather Temperature.png')
  plt.show()
    
GEN_weather_crimes(reports_weather) 

### FIXED rate of overall crime per hour (stacked according to PdDistrict)

def GEN_crime_byHour(originaldata):
  crimeByHour = originaldata.loc[(originaldata.PdDistrict.isna() == False), :]
  crimeByHour = crimeByHour.groupby(["Hour", "PdDistrict"]).size()
  data = crimeByHour.unstack(level = 1)

  hf = [0]*len(data.index)
  for i in range(len(data.columns)):
    hf += data[data.columns[i]]

  data['HourFreq'] = pd.Series(hf)

  data = data.sort_values(by = 'HourFreq', ascending = False).drop('HourFreq', axis = 1)
  data.loc['Total'] = data.sum()
  weightOrder_data = data.sort_values(by = "Total", axis = 1, ascending = False).drop('Total', axis = 0)
  weightOrder_data[weightOrder_data.columns].plot.bar(stacked = True)

  plt.xlabel('Hour')
  plt.ylabel('Crime Frequency')
  plt.title('Reported Incidents by Hour', fontweight = 'bold')
  plt.legend(loc='upper center', bbox_to_anchor=(1.17, 0.8), shadow=True, ncol=1)
  plt.savefig('Reported Incidents by Hour.png', dpi = 300)
  plt.show()

GEN_crime_byHour(reports2003_present_Q2)

#%%
### K-MEANS CLUSTERING FOR ADDING INCOME (in thousands)
# look first year of 2018
dist_income = [['INGLESIDE', 81700], ['SOUTHERN', 117800], ['CENTRAL', 113600], ['BAYVIEW', 57500], ['RICHMOND', 85600], ['TARAVAL', 106800], ['TENDERLOIN', 27400], ['MISSION', 96300], ['NORTHERN', 43800], ['PARK', 106900]]
dfSFIncome = pd.DataFrame(dist_income)
dfSFIncome.columns = ['district', 'income']
dfSFIncome = dfSFIncome.set_index(['district'])

# Data Preparation -- one row per district!
### each row should contain the percent of each crime type
reports2003_presentC = reports2003_present.copy()
reports2003_presentC = reports2003_presentC.loc[(reports2003_presentC.Year == 2018) & ~(reports2003_presentC.PdDistrict.isna()), ['PdDistrict', 'Category']]
top_ten_crimes = reports2003_presentC.groupby(['Category']).size().reset_index().rename(columns = {0:'freq'}).nlargest(10, ['freq'])

# data frame contains all reported incidents with top 10 crimes of 2018
top_ten_2018 = reports2003_presentC.loc[(reports2003_presentC.Category == top_ten_crimes.Category.iloc[0]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[1]) |
        (reports2003_presentC.Category == top_ten_crimes.Category.iloc[2]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[3]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[4]) |
        (reports2003_presentC.Category == top_ten_crimes.Category.iloc[5]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[6]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[7]) |
        (reports2003_presentC.Category == top_ten_crimes.Category.iloc[8]) | (reports2003_presentC.Category == top_ten_crimes.Category.iloc[9]), :]

top_ten_2018['counter'] = 1
grouped = top_ten_2018.groupby(['PdDistrict', 'Category'])['counter'].sum() # returns a series
# data frame contains number of each of the top 10 crimes per district
grouped_df = grouped.to_frame().reset_index()

assert grouped_df.loc[:, ['PdDistrict', 'Category']].duplicated().value_counts().shape[0] == 1

newGrouped = (pd.merge(grouped_df
                      , grouped_df.groupby('PdDistrict', as_index = False).agg({'counter':'sum'}).rename(columns = {'counter':'total_crimes_in_dist'})
                      , how = 'inner', on = 'PdDistrict')
             )

# generate the percent

# generate the percent for each crime type
newGrouped.loc[:, 'counter_pct'] = newGrouped.loc[:, 'counter']/newGrouped.loc[:, 'total_crimes_in_dist'] 
#newdata = newGrouped.unstack(level = 0)

# reshape using unstack
newGrouped = newGrouped.loc[:, ['PdDistrict', 'Category', 'counter_pct']].set_index(['PdDistrict', 'Category']).unstack('Category')

# remove multiindex
newGrouped.columns = newGrouped.columns.droplevel()


#%%

# clustering model
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(newGrouped)
res_labels = pd.DataFrame(kmeans.labels_)
res_labels.index = newGrouped.index
res_labels.columns = ['kmeans_group']
# sum of squares
print('Inertia : ', kmeans.inertia_)

df_combined = (pd
               .merge(newGrouped, dfSFIncome, right_index=True, left_index=True)
               .merge(res_labels, right_index=True, left_index=True)
              )

df_combined.head()
df_combined.kmeans_group.value_counts()
incomeGroup = df_combined.loc[:, ['income', 'kmeans_group']].sort_values('income', ascending=False)
print(incomeGroup)

# figured out what were the high income districts from observing percentage distribution of reported crimes
# did accurate job for six figure districts

# perform k-means clustering with multiple clusters
all_cluster_data = []

for n_cluster in range(1, 10):
    kmeans_t = KMeans(n_clusters= n_cluster, random_state=0).fit(newGrouped)
    res_labels_t = pd.DataFrame(kmeans_t.labels_)
    res_labels_t.index = newGrouped.index
    res_labels_t.columns = ['kmeans_group']
    
    df_combined_t = pd.merge( newGrouped, dfSFIncome, right_index=True, left_index=True).merge(res_labels_t, right_index=True, left_index=True)
    all_cluster_data.append([n_cluster, kmeans_t.inertia_, df_combined_t.copy()])

    
to_plot_df = pd.DataFrame([[x[0], x[1]] for x in all_cluster_data], columns=['clusters', 'sumsquares'])

plt.bar(to_plot_df.clusters, to_plot_df.sumsquares)
plt.xlabel('clusters')
plt.ylabel('sumsquares')
plt.title('K-Means Clustering', fontweight = 'bold')
plt.savefig('K-Means Clustering Elbow Method.png')
plt.show()


#%%

### QUESTION 2 EXPLANATORY LINEAR REGRESSION MODEL
reports2003_present_Q2 = reports2003_present.loc[:, ['Category', 'Date', 'Hour', 'X', 'Y', 'PdDistrict']]
weather2003_Q2 = weather2003.loc[:,['Date', 'Hour', 'AirTemp']]
reports_weather = pd.merge(reports2003_present_Q2, weather2003_Q2, how = 'left', on = ['Date', 'Hour'])
model_flag = 1

def Processing_data(reports_weather, model_flag):
    ####Setup
#    reports_weather = pd.merge(data1, data2, how = 'left', on = ['Date', 'Hour'])
    # convert day of week to integer (Monday = 0...Sunday = 6)
    reports_weather['DOW'] = pd.to_datetime(reports_weather['Date']).dt.dayofweek
    
    # merged df with only non-null air temperatures
    reports_weather_fix = reports_weather.loc[(reports_weather.AirTemp.isna() == False), :].reset_index()
    reports_weather_fix = reports_weather_fix.drop('index', axis =1)
    
    # 52 unique crime categories
    # look for 10 most popular crime categories
    topCrimeTypes = reports_weather_fix.groupby(['Category']).size().reset_index().rename(columns = {0:'freq'}).nlargest(10, ['freq'])
    top_ten_crimes = topCrimeTypes.Category
    # df containing all reported incidents with 10 most frequent crime categories
    top_ten_over_yrs = reports_weather_fix.loc[(reports_weather_fix.Category == top_ten_crimes.iloc[0]) | (reports_weather_fix.Category == top_ten_crimes.iloc[1]) |
            (reports_weather_fix.Category == top_ten_crimes.iloc[2]) | (reports_weather_fix.Category == top_ten_crimes.iloc[3]) | (reports_weather_fix.Category == top_ten_crimes.iloc[4]) |
            (reports_weather_fix.Category == top_ten_crimes.iloc[5]) | (reports_weather_fix.Category == top_ten_crimes.iloc[6]) | (reports_weather_fix.Category == top_ten_crimes.iloc[7]) |
            (reports_weather_fix.Category == top_ten_crimes.iloc[8]) | (reports_weather_fix.Category == top_ten_crimes.iloc[9]), :]
    
    # define undersample strategy
    
    
    # data frame with no null (missing) coordinates
    # assuming that recorded coordinates at the location of crime suggest more accurate PdDistricts (our definition of area)
    allCoordinates = top_ten_over_yrs.loc[(top_ten_over_yrs.X.isna() == False) | (top_ten_over_yrs.Y.isna() == False), :]
    
    # extract more features for prediction purposes
    # dayClass (weekend or weekday)
    allCoordinates.loc[(allCoordinates.DOW == 0)|(allCoordinates.DOW == 1)|(allCoordinates.DOW == 2)|(allCoordinates.DOW == 3)|(allCoordinates.DOW == 4), 'dayClass'] = 'Weekday'
    allCoordinates.loc[(allCoordinates.DOW == 5)|(allCoordinates.DOW == 6), 'dayClass'] = 'Weekend'
    
    allCoordinates['Date'] = pd.to_datetime(allCoordinates['Date'])
    # season
    allCoordinates.loc[(allCoordinates.Date.dt.month == 12) | (allCoordinates.Date.dt.month == 1) | (allCoordinates.Date.dt.month == 2), 'Season'] = 'Winter'
    allCoordinates.loc[(allCoordinates.Date.dt.month == 3) | (allCoordinates.Date.dt.month == 4) | (allCoordinates.Date.dt.month == 5), 'Season'] = 'Spring'
    allCoordinates.loc[(allCoordinates.Date.dt.month == 6) | (allCoordinates.Date.dt.month == 7) | (allCoordinates.Date.dt.month == 8), 'Season'] = 'Summer'
    allCoordinates.loc[((allCoordinates.Date.dt.month >= 8) & (allCoordinates.Date.dt.month < 12)), 'Season'] = 'Fall'
    
    # adjusting for daylight savings
    allCoordinates.loc[((allCoordinates.Season == 'Spring') | (allCoordinates.Season == 'Summer') | (allCoordinates.Season == 'Fall')), 'DST'] = 1 # DST begins
    allCoordinates.loc[(allCoordinates.Season == 'Winter'), 'DST'] = 0 # DST ends
    
    allCoordinates.loc[(allCoordinates.DST == 1) & ((allCoordinates.Hour > 5) & (allCoordinates.Hour < 11)), 'Time'] = 'Morning'
    allCoordinates.loc[(allCoordinates.DST == 1) & ((allCoordinates.Hour >= 11) & (allCoordinates.Hour < 17)), 'Time'] = 'Afternoon'
    allCoordinates.loc[(allCoordinates.DST == 1) & ((allCoordinates.Hour >= 17) & (allCoordinates.Hour < 21)), 'Time'] = 'Evening'
    allCoordinates.loc[(allCoordinates.DST == 1) & (((allCoordinates.Hour >= 21) & (allCoordinates.Hour <= 23)) | ((allCoordinates.Hour > 0) & (allCoordinates.Hour <= 5)) | (allCoordinates.Hour == 0)), 'Time'] = 'Night'
    
    # for non-DST months
    allCoordinates.loc[(allCoordinates.DST == 0) & ((allCoordinates.Hour > 6) & (allCoordinates.Hour < 11)), 'Time'] = 'Morning'
    allCoordinates.loc[(allCoordinates.DST == 0) & ((allCoordinates.Hour >= 11) & (allCoordinates.Hour < 16)), 'Time'] = 'Afternoon'
    allCoordinates.loc[(allCoordinates.DST == 0) & ((allCoordinates.Hour >= 16) & (allCoordinates.Hour < 20)), 'Time'] = 'Evening'
    allCoordinates.loc[(allCoordinates.DST == 0) & (((allCoordinates.Hour >= 20) & (allCoordinates.Hour <= 23)) | ((allCoordinates.Hour > 0) & (allCoordinates.Hour <= 6)) | (allCoordinates.Hour == 0)), 'Time'] = 'Night'
    
    
    # downtown vs residential
    allCoordinates.loc[(allCoordinates.PdDistrict == 'NORTHERN') | (allCoordinates.PdDistrict == 'CENTRAL') | (allCoordinates.PdDistrict == 'SOUTHERN') | (allCoordinates.PdDistrict == 'TENDERLOIN') | (allCoordinates.PdDistrict == 'BAYVIEW'), 'distType'] = 'downtown'
    allCoordinates.loc[(allCoordinates.PdDistrict == 'RICHMOND') | (allCoordinates.PdDistrict == 'TARAVAL') | (allCoordinates.PdDistrict == 'MISSION') | (allCoordinates.PdDistrict == 'PARK') | (allCoordinates.PdDistrict == 'INGLESIDE'), 'distType'] = 'residential'
    
    #categorizing air temp
    allCoordinates.loc[(allCoordinates.AirTemp < 11.67), 'tempClass'] = 'cold'
    allCoordinates.loc[(allCoordinates.AirTemp >= 11.67) & (allCoordinates.AirTemp < 21), 'tempClass'] = 'avg temp'
    allCoordinates.loc[(allCoordinates.AirTemp > 21), 'tempClass'] = 'hot'
    
    #residential and downtown at night
    allCoordinates = allCoordinates.assign(res_night = 0)
    allCoordinates = allCoordinates.assign(dtwn_night = 0)
    allCoordinates.loc[(allCoordinates.distType == 'residential') & (allCoordinates.Time == 'Night'), 'res_night'] = 1
    allCoordinates.loc[(allCoordinates.distType == 'downtown') & (allCoordinates.Time == 'Night'), 'dtwn_night'] = 1
       
    
    if (model_flag == 1):
        Pd_oneHot = pd.get_dummies(allCoordinates['PdDistrict'], drop_first = True)
        DOW_oneHot = pd.get_dummies(allCoordinates['dayClass'], drop_first = True)
        time_oneHot = pd.get_dummies(allCoordinates['Time'], drop_first = True)
        season_oneHot = pd.get_dummies(allCoordinates['Season'], drop_first = True)
        dist_oneHot = pd.get_dummies(allCoordinates['distType'], drop_first = True)
        temp_oneHot = pd.get_dummies(allCoordinates['tempClass'], drop_first = True)
    else:
        Pd_oneHot = pd.get_dummies(allCoordinates['PdDistrict'])
        DOW_oneHot = pd.get_dummies(allCoordinates['dayClass'])
        time_oneHot = pd.get_dummies(allCoordinates['Time'])
        season_oneHot = pd.get_dummies(allCoordinates['Season'])
        dist_oneHot = pd.get_dummies(allCoordinates['distType'])
        temp_oneHot = pd.get_dummies(allCoordinates['tempClass'])
        

    allCoordinates = allCoordinates.join(Pd_oneHot)
    allCoordinates = allCoordinates.join(DOW_oneHot)
    allCoordinates = allCoordinates.join(time_oneHot)
    allCoordinates = allCoordinates.join(season_oneHot)
    allCoordinates = allCoordinates.join(dist_oneHot)
    allCoordinates = allCoordinates.join(temp_oneHot)
 
    allCoordinates = allCoordinates.drop(['PdDistrict', 'DOW', 'dayClass', 'Time', 'Season', 'distType', 'tempClass', 'DST'], axis = 1)

    freq = allCoordinates.groupby(['Category']).size().reset_index().rename(columns = {0:'freq'})
    allCoordinates = pd.merge(allCoordinates, freq, on = 'Category')
    
    allCoordinates['percent'] = allCoordinates['freq']/allCoordinates.freq.sum()
    allCoordinates = allCoordinates.drop('freq', axis = 1)
            
    return allCoordinates

Q2_data = Processing_data(reports_weather, model_flag)

#%%

def Linear(originaldata):
    print("=======LINEAR REGRESSION=======")
    X_orig = originaldata.loc[:, originaldata.columns != 'percent']
    X_orig = X_orig.iloc[:, 6:]
    X = X_orig.values
    Y = originaldata['percent'].values
    
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size = .2)
    
    # focus on crime in each district
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    print('Coefficients: ', lm.coef_)
    coeff_df = pd.DataFrame(lm.coef_, X_orig.columns, columns = ['Coefficient'])
    print(coeff_df)

coeff_df = Linear(Q2_data)

#%%
#################################QUESTION 3 IN FUNCTIONS###############################
reports2003_present_Q3 = reports2003_present.loc[:, ['Category', 'Date', 'Year', 'Hour', 'X', 'Y', 'PdDistrict']]
weather2003_Q3 = weather2003.loc[:,['Date', 'Hour', 'AirTemp']]
reports_weather = pd.merge(reports2003_present_Q3, weather2003_Q3, how = 'left', on = ['Date', 'Hour'])
model_flag = 0
Q3_data = Processing_data(reports_weather, model_flag)    
Q3_data = Q3_data.drop('percent', axis = 1)  
Q3_data['Category'] = Q3_data['Category'].astype('category')
# assign the encoded variable to a new column using the cat.codes accessor
Q3_data['Category_cat'] = Q3_data['Category'].cat.codes
#Q3_data.head()


def GEN_NeuralNetwork(originaldata):
    print("=======NeuralNetwork=======")
    Y = originaldata['Category_cat']
    X = originaldata.drop('Category_cat', axis = 1)
    X = X.iloc[:, 7:]

    # split data into train and test set via sklearn
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size = .2)
    
    
    X_train = X_train.values
    X_test = X_test.values
    X_normTrain = (X_train - X_train.mean(axis = 0))/X_train.std(axis = 0)
    X_normTest = (X_test - X_train.mean(axis = 0))/X_train.std(axis = 0)
    
    
    # perform one-hot encoding on training labels (crime categories))
    Y_oneHotTrain = pd.get_dummies(Y_train).values
    Y_oneHotTest = pd.get_dummies(Y_test).values
    
    # create validation set
    # take training data and partition it into k equal sized chunks
    # in this example, k = 5
#    X_val = X_normTrain[:41255]
#    Y_val = Y_oneHotTrain[:41255]
#    partial_X_train = X_normTrain[41255:]Q
#    partial_Y_train = Y_oneHotTrain[41255:]
    
    # split train and validation data
    X_train_reduced, X_val, Y_train_reduced, Y_val = train_test_split(X_normTrain, Y_oneHotTrain, test_size = .2)
    
    # neural network model    
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(80, activation = 'relu', input_shape = (27,)),
            tf.keras.layers.Dense(80, activation = 'relu'),
            tf.keras.layers.Dense(10, activation = 'softmax')])
    
    # cross entropy loss function
    # to measure agreement btwn predicted prob. vector and ground truth prob. vector
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #numEpochs = 17
    numEpochs = 7
    #history = model.fit(X_train, Y_train, batch_size = 600, epochs = numEpochs)
    history = model.fit(X_train_reduced, Y_train_reduced, batch_size = 100, epochs = numEpochs, validation_data = (X_val, Y_val))
    history_dict = history.history
    
    
    ######Graph
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    train_accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    epochs = np.arange(1, numEpochs+1)
    
    # for interactive zoom of plots -> %matplotlib qt5
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Crimes Training and Validation Loss.png')
    
    # signs of overfitting at 7 epochs
    
    plt.figure()
    plt.plot(epochs, train_accuracy, 'bo', label = 'Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Crimes Training and Validation Accuracy.png')
    
    # observe how we did on test data...
    Y_pred = model.predict(X_normTest)
    # perform one-hot encoding on predicted labels
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    # 
    Y_observed = np.argmax(Y_oneHotTest, axis = 1)
    

    # perform one-hot encoding on predicted test labels
    # simply put, return index with maximum probability

        
    # after 6 epochs, classification accuracy no longer improves on validaton set 
    return Y_pred_classes, Y_observed
Y_pred_classes, Y_observed = GEN_NeuralNetwork(Q3_data)


### QUESTION 3 OWN IMPLEMENTATION OF MULTICLASS LOGISTIC REGRESSION w/SGD

# softmax function
# applies exponential function to each component of u

def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

def crossEntropy(y, q):
    return -np.vdot(y, np.log(q))


def eval_L(X, Y, beta): # evalutes cost function L(beta)
     # assume feature vectors are augmented already
     
     N = X.shape[0] # number of training examples
     L = 0.0
     
     # visit each training example one at a time
     for i in range(N):
         XiHat = X[i] # ith feature vector 
         Yi = Y[i] # corresponding traning label
         
         qi = softmax(beta @ XiHat)
         
         # loss for ith training example
         L += crossEntropy(Yi, qi)
         
     return L
         

# epoch: 1 full sweep through entire training dataset
def logReg_SGD(X, Y, alpha):
    numEpochs = 5
    print("Alpha: " + str(alpha))
    print("Stochastic Gradient Descent")
    
    N, d = X.shape
    # augment feature vectors
    #X = np.insert(X, 0, 1, axis = 1)
    
    K = Y.shape[1]
    beta = np.zeros((K, d))
    
    # store the progress of L(beta) per epoch
    listLvals = []
    
    # for each epoch, sweep thru training dataset in random order
    for ep in range(numEpochs):
        # begin 1 epoch of SGD
        L = eval_L(X, Y, beta)
        
        # cost function values should be decreasing
        # since our goal is to minimize cost function L(beta)
        listLvals.append(L)
        
        print("Current epoch: " + str(ep) + " Cost is: " + str(L))
        
                
        # randomly shuffle training data
        prm  = np.random.permutation(N) 
        
        # visiting each training example in randomized order
        for i in prm:
            XiHat = X[i] # ith feature vector
            Yi = Y[i] # ith row of ground truth probability vector
            
            # create predicted probability vector
            # computes all K dot products at once
            qi = softmax(beta @ XiHat) # multiply beta matrix by vector XiHat
            
            # compute gradient of L_i(beta)
            grad_Li = np.outer(qi - Yi, XiHat)
            beta = beta - alpha * grad_Li
    
    # measure if algorithm is making progress
    # value of L(beta) aka cost function should be reduced
    return beta, listLvals

def predictLabels(X, beta):

    N = X.shape[0]
    
    predictions = []
    probabilities = []
    
    for i in range(N):
        XiHat = X[i]
        # create predicted probability vector
        # computes all K dot products at once
        qi = softmax(beta @ XiHat)
        
        # predicted class
        k = np.argmax(qi)
        predictions.append(k)
        probabilities.append(np.max(qi))
    return predictions, probabilities


def check(predictions, Y_oneHotTest, numTest):
    # check number of correct predictions
    numCorrect = 0
    for i in range(numTest):
        if predictions[i] == np.argmax(Y_oneHotTest[i]):
            numCorrect += 1
            
    accuracy = numCorrect/numTest
    score = print('Accuracy: ' + str(accuracy))
    
    return score


def GEN_SGD(originaldata):
    print("=======SGD=======")
    Y = originaldata['Category_cat']
    X = originaldata.drop('Category_cat', axis = 1)
    X = X.iloc[:, 7:]
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size = .2)
    
    X_normTrain = (X_train - X_train.mean(axis = 0))/X_train.std(axis = 0)
    X_normTest = (X_test - X_train.mean(axis = 0))/X_train.std(axis = 0)
    
    numTrain = X_normTrain.shape[0]
    numTest = X_normTest.shape[0]
    # perform one-hot encoding on training labels (crime categories)) -> top 15 crimes
    Y_oneHotTrain = pd.get_dummies(Y_train).values
    Y_oneHotTest = pd.get_dummies(Y_test).values
    
    # augment feature vectors
    #N, d = X_normTrain.shape # N = 245961, d = 25
    allOnesTrain = np.ones((numTrain, 1))
    X_train = np.hstack((allOnesTrain, X_normTrain))
    allOnesTest = np.ones((numTest, 1))
    X_test = np.hstack((allOnesTest, X_normTest))
    Y_oneHotTest = pd.get_dummies(Y_test).values
    
    alpha = .00001 # learning rate
    #beta, listLvals = logReg_GD(X_train, Y_oneHot, alpha)  
    #predictions, probabilities = predictLabels(X_test, beta)   
    beta, listLvals = logReg_SGD(X_train, Y_oneHotTrain, alpha)  

    predictions, probabilities = predictLabels(X_test, beta) 
    score = check(predictions, Y_oneHotTest, numTest)  
    # semilogy plot that shows cost function value vs. iteration
    plt.figure()
    plt.semilogy(listLvals, 'b', label = 'Alpha = .00001')
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost Function Value')
    plt.title('Cost Function Value per Iteration of GD')
    plt.legend()
    plt.show()
    
    return predictions, probabilities

# 0.4192804499914302 accuracy with newly created features; slightly higher than NN
    
predictions, probabilities = GEN_SGD(Q3_data)
