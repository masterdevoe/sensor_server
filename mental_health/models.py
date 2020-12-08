#########################################################################################################
class Depression:
  import pickle
  import pandas as pd
  import numpy as np
  import datetime
  from datetime import datetime as dt
  import statistics as st
  import math as mt
  from scipy import signal
  from sklearn.cluster import KMeans
  # from numpy import diff
  # from sklearn.metrics import silhouette_score
  # from sklearn.metrics import pairwise_distances
  # from scipy.signal import argrelextrema
  from kneed import KneeLocator
  #UTC Timestamp to Datetime.
  def utc_to_ts(self,x):
    return(self.dt.utcfromtimestamp(x))

  def __init__(self,csv_data,model_path):
    #self.csv_path=csv_path
    #self.df=self.pd.read_csv(csv_path)

    file=open(model_path,"rb")
    self.model=self.pickle.load(file)
    file.close()

    self.df=csv_data

    #TBR
    # columns=self.df.columns
    # self.df=self.df.drop(columns="travelstate")
    # self.df=self.df.fillna(0)
    # self.df=self.df.reset_index()
    # self.df.columns=columns
    self.df["time"]=self.df["time"].apply(self.utc_to_ts)

  #Returns location varience for a dataframe.
  def loc_varience(self):
    x=self.df
    var_long=self.st.variance(x.longitude)
    var_lat=self.st.variance(x.latitude)
    loc_var=self.mt.log(var_long+var_lat)
    return(loc_var)
  
  #Returns speed mean for a dataframe.
  def speed_mean(self):
    x=self.df
    time=x.time
    lat=x.latitude
    lon=x.longitude
    time_list=[(time[i+1]-time[i]).seconds for i in range(len(time)-1)]
    lat_list=[lat[i+1]-lat[i] for i in range(len(lat)-1)]
    lon_list=[lon[i+1]-lon[i] for i in range(len(lon)-1)]
    inst_speed=[((lat_list[i]/time_list[i])**2 + (lon_list[i]/time_list[i])**2)**0.5 for i in range(len(time_list))]
    speed_mean=self.st.mean(inst_speed)
    return(speed_mean)

  #Returns speed varience for a dataframe.
  def speed_variance(self):
    x=self.df
    time=x.time
    lat=x.latitude
    lon=x.longitude
    time_list=[(time[i+1]-time[i]).seconds for i in range(len(time)-1)]
    lat_list=[lat[i+1]-lat[i] for i in range(len(lat)-1)]
    lon_list=[lon[i+1]-lon[i] for i in range(len(lon)-1)]
    inst_speed=[((lat_list[i]/time_list[i])**2 + (lon_list[i]/time_list[i])**2)**0.5 for i in range(len(time_list))]
    speed_variance=self.st.variance(inst_speed)
    return(speed_variance)

  #Returns total for a dataframe.
  def total_distance(self):
    x=self.df
    lat=x.latitude
    lon=x.longitude
    lat_list=[(lat[i+1]-lat[i])**2 for i in range(len(lat)-1)]
    lon_list=[(lon[i+1]-lon[i])**2 for i in range(len(lon)-1)]
    total_distance=(sum(lat_list)+sum(lon_list))**0.5
    return(total_distance)

  #Returns Circadian Movement for a dataframe.
  def circadian_movement(self):
    df=self.df
    _, psd_lat = self.signal.welch(df.latitude)
    _, psd_long = self.signal.welch(df.longitude)
    E_lat=self.st.mean(psd_lat)
    E_lon=self.st.mean(psd_long)
    circadian=self.mt.log(E_lat+E_lon)
    return(circadian)

  #Returns optimal location clusters for a dataframe.
  def location_clusters(self):
    df=self.df
    K_clusters = range(2,10)
    #N_cl=[i for i in K_clusters]
    #hdb=[hdbscan.HDBSCAN(min_cluster_size=i, metric='haversine',core_dist_n_jobs=-1) for i in K_clusters]
    kmeans = [self.KMeans(n_clusters=i,max_iter=500,n_jobs=-1,algorithm='full') for i in K_clusters]
    dataset=df
    # Y_axis = dataset['latitude']
    # X_axis = dataset['longitude']
    coordinates=[(dataset.loc[i,"latitude"],dataset.loc[i,"longitude"]) for i in range(len(dataset))]
    #cluster_labels=[i.fit_predict(coordinates)  for i in hdb]
    cluster_labels=[i.fit(coordinates)  for i in kmeans]
    #score = [silhouette_score(coordinates,i,metric='haversine') for i in cluster_labels]
    score=[i.inertia_ for i in cluster_labels]
    x = range(2, len(score)+2)
    kn = self.KneeLocator(x, score, curve='convex', direction='decreasing')
    return(kn.knee)

  #Returns entropy for a dataframe.
  def entropy(self,n_clusters):
    df=self.df
    cluster=n_clusters
    clusterModel=self.KMeans(n_clusters=cluster,max_iter=500,n_jobs=-1,algorithm='full')
    dataset=df
    # Y_axis = dataset['latitude']
    # X_axis = dataset['longitude']
    coordinates=[(dataset.loc[i,"latitude"],dataset.loc[i,"longitude"]) for i in range(len(dataset))]
    clusterModel.fit(coordinates)
    clusters=clusterModel.predict(coordinates)
    counts=self.pd.Series(clusters).value_counts()
    percentages=[i/sum(counts) for i in list(counts)]
    entropy=-sum([i*self.mt.log(i) for i in percentages])
    return(entropy)  

  #returns normalised entropy for a entropy and n_clusters.
  normalised_entropy=lambda self,entropy,n_clusters: entropy/self.mt.log(n_clusters)        
 #returns raw entropy for a datframe.
  def raw_entropy(self):
    df=self.df
    n=10
    model=self.KMeans(n_clusters=n,n_jobs=-1,max_iter=500,algorithm='full')
    dataset=df
    # Y_axis = dataset['latitude']
    # X_axis = dataset['longitude']
    coordinates=[(dataset.loc[i,"latitude"],dataset.loc[i,"longitude"]) for i in range(len(dataset))]
    clusters=model.fit_predict(coordinates)
    counts=self.pd.Series(clusters).value_counts()
    percentages=[i/sum(counts) for i in list(counts)]
    RawEntropy=-sum([i*self.mt.log(i) for i in percentages])
    return(RawEntropy)

  #To make feature calculation and final prediction.
  def predict(self):
    loc_var=self.loc_varience()
    spd_mean=self.speed_mean()
    spd_var=self.speed_variance()
    tot_dist=self.total_distance()
    circ_movement=self.circadian_movement()
    loc_clusters=self.location_clusters()
    entrpy=self.entropy(loc_clusters)
    norm_entropy=self.normalised_entropy(entrpy,loc_clusters)
    raw_entrpy=self.raw_entropy()

    order=[loc_var,spd_mean,spd_var,tot_dist,circ_movement,loc_clusters,entrpy,norm_entropy,raw_entrpy]

    return(self.model.predict(order)[0])
############################################################################################################

class Bipolar:
  import numpy as np
  import pandas as pd
  import pickle
  from scipy import signal
  import math as mt
  import statistics as st
  from scipy import stats as sc
  import datetime
  import os
  import random

  def get_data(self,path):
    df=self.pd.read_csv(path)
    df.timestamp=df.timestamp.astype("datetime64")
    df.date=df.date.astype("datetime64")
    return(df)

  def __init__(self,csv_data,model_path):
    file=open(model_path,"rb")
    self.model=self.pickle.load(file)
    file.close()
    #self.df=self.get_data(csv_path)
    self.df=csv_data

  #Mean
  def mean(self):
    df=self.df
    return(self.st.mean(df.activity)) 
  #Standard Deviation
  def stdev(self):
    df=self.df
    return(self.st.stdev(df.activity))
  #Maximum value
  def maximum(self):
    df=self.df
    return(max(df.activity))
  #Minimum Value
  def minimum(self):
    df=self.df
    return(min(df.activity))      
  #Median
  def median(self):
    df=self.df
    return(self.st.median(list(df.activity)))  
  #interquartile range
  def IQR(self):
    df=self.df
    return(self.np.subtract(*self.np.percentile(df.activity, [75, 25])))   
  #skewness
  def skew(self):
    return(self.sc.skew(self.df.activity)) 
  #kurtosis
  def kurt(self):
    df=self.df
    return(self.sc.kurtosis(df.activity))
  #root mean square
  def rms(self):
    df=self.df
    a=list(df.activity)
    a_sq=[i**2 for i in a]
    ms=self.st.mean(a_sq)
    return(ms**0.5)
  #Energy
  def energy(self):
    df=self.df
    a=list(df.activity)
    a_sq=[i**2 for i in a]
    return(sum(a_sq))
  #SRA
  def SRA(self):
    df=self.df
    mu=self.mean()
    sqrt_abs_x=[abs(i)**0.5 for i in df.activity]
    return(mu*sum(sqrt_abs_x))
  #Peak to Peak
  def PTP(self):
    return(self.maximum()-self.minimum())

  #Impulse Factor
  def IMP(self):
    return(self.maximum()/self.mean())

  #Crest Factor
  def CF(self):
    return(self.maximum()/self.rms())

  def logpsd(self):
    df=self.df
    _,psd=self.signal.welch(df.activity)
    return(self.mt.log(self.st.mean(psd)))            

  def predict(self):
    Mean=self.mean()
    Stdev=self.stdev()
    Median=self.median()
    iqr=self.IQR()
    Skewness=self.skew()
    Kurtosis=self.kurt()
    RMS=self.rms()
    Energy=self.energy()
    sra=self.SRA()
    ptp=self.PTP()
    imp=self.IMP()
    cf=self.CF()
    Logpsd=self.logpsd()

    order=[Mean,Stdev,Median,iqr,Skewness,Kurtosis,RMS,Energy,sra,ptp,imp,cf,Logpsd]

    return(self.model.predict(order)[0])
#####################################################################################################
class Sleep_hours:
  import pandas as pd
  import statistics as st
  import pickle
  from datetime import datetime

  def to_datetime(self,x):
    return(self.datetime.fromtimestamp(x))

  def __init__(self, dark_data, activity_data, gps_data, phonelock_data, audio_inference, model_file):

    # self.dark_data=       self.pd.read_csv(dark_data)
    # self.activity_data=   self.pd.read_csv(activity_data)
    # self.gps_data=        self.pd.read_csv(gps_data)
    # self.phonelock_data=  self.pd.read_csv(phonelock_data)
    # self.audio_inference= self.pd.read_csv(audio_inference)

    self.dark_data=       dark_data
    self.activity_data=   activity_data
    self.gps_data=        gps_data
    self.phonelock_data=  phonelock_data
    self.audio_inference= audio_inference
    
    file=open(model_file,"rb")
    self.model=self.pickle.load(file)
    file.close()


  #Percentage of Still activity
  def acti(self): #x is dataframe having activity inference for one day with column activity inference
    x=self.activity_data
    x.dropna(how='any', inplace=True)
    counts=x[' activity inference'].value_counts(normalize=True)  
    return(counts[0])
  #Location variance
  def locvar(self): # x is a df with columns: timestamp, latitude, longitude
    x=self.gps_data
    var_lat=self.st.variance(x.longitude)
    var_long=self.st.variance(x.latitude)
    loc_var= (var_lat+var_long)**(1/2)
    return(loc_var)  
  #Max phonelock time
  def phonelock(self):
    self.phonelock_data.start=self.phonelock_data.start.apply(self.to_datetime)
    self.phonelock_data.end=self.phonelock_data.end.apply(self.to_datetime)
    x=self.phonelock_data 
    time=list(x.end-x.start)
    time=[i.total_seconds()/60 for i in time]
    max_pl=max(time)
    return(max_pl)
  #Max dark time: # has two columns start time and end time
  def darktime(self):
    self.dark_data.start=self.dark_data.start.apply(self.to_datetime)
    self.dark_data.end=self.dark_data.end.apply(self.to_datetime)
    x=self.dark_data
    time=list(x.end-x.start)
    time=[i.total_seconds()/60 for i in time]
    max_dk=max(time)
    return(max_dk)

  #Percentage of no noise:

  def audio(self): #audio inference file is used where the value 0 means no background noise
    x=self.audio_inference
    audio=x[" audio inference"].value_counts(normalize=True)
    return(audio[0])

  def predict(self):
    dark=self.darktime()
    loc_var=self.locvar()
    act=self.acti()
    phone=self.phonelock()
    audio_inf=self.audio()

    order=[dark, loc_var, act, phone, audio_inf]    
    return(self.model.predict([order])[0])
#################################################################################################  

class Sleep_sqi:
  import pandas as pd
  import statistics as st
  import pickle
  from datetime import datetime

  def to_datetime(self,x):
    return(self.datetime.fromtimestamp(x))

  def __init__(self, dark_data, activity_data, gps_data, phonelock_data, audio_inference, model_file):

    # self.dark_data=       self.pd.read_csv(dark_data)
    # self.activity_data=   self.pd.read_csv(activity_data)
    # self.gps_data=        self.pd.read_csv(gps_data)
    # self.phonelock_data=  self.pd.read_csv(phonelock_data)
    # self.audio_inference= self.pd.read_csv(audio_inference)

    self.dark_data=       dark_data
    self.activity_data=   activity_data
    self.gps_data=        gps_data
    self.phonelock_data=  phonelock_data
    self.audio_inference= audio_inference
    
    file=open(model_file,"rb")
    self.model=self.pickle.load(file)
    file.close()


  #Percentage of Still activity
  def acti(self): #x is dataframe having activity inference for one day with column activity inference
    x=self.activity_data
    x.dropna(how='any', inplace=True)
    counts=x[' activity inference'].value_counts(normalize=True)  
    return(counts[0])
  #Location variance
  def locvar(self): # x is a df with columns: timestamp, latitude, longitude
    x=self.gps_data
    var_lat=self.st.variance(x.longitude)
    var_long=self.st.variance(x.latitude)
    loc_var= (var_lat+var_long)**(1/2)
    return(loc_var)  
  #Max phonelock time
  def phonelock(self):
    self.phonelock_data.start=self.phonelock_data.start.apply(self.to_datetime)
    self.phonelock_data.end=self.phonelock_data.end.apply(self.to_datetime)
    x=self.phonelock_data 
    time=list(x.end-x.start)
    time=[i.total_seconds()/60 for i in time]
    max_pl=max(time)
    return(max_pl)
  #Max dark time: # has two columns start time and end time
  def darktime(self):
    self.dark_data.start=self.dark_data.start.apply(self.to_datetime)
    self.dark_data.end=self.dark_data.end.apply(self.to_datetime)
    x=self.dark_data
    time=list(x.end-x.start)
    time=[i.total_seconds()/60 for i in time]
    max_dk=max(time)
    return(max_dk)

  #Percentage of no noise:

  def audio(self): #audio inference file is used where the value 0 means no background noise
    x=self.audio_inference
    audio=x[" audio inference"].value_counts(normalize=True)
    return(audio[0])

  def predict(self):
    dark=self.darktime()
    loc_var=self.locvar()
    act=self.acti()
    phone=self.phonelock()
    audio_inf=self.audio()

    order=[dark, loc_var, act, phone, audio_inf]    
    return(self.model.predict([order])[0])
#######################################################################################################################
class Stress:
  import pickle
  import pandas as pd
  import numpy as np
  import statistics as st
  import math as mt
  import hdbscan
  def __init__(self,csv_data,model_path):

    file=open(model_path,"rb")
    self.model=self.pickle.load(file)
    file.close()

    #self.tstdata=self.pd.read_csv(csv_path)
    self.tstdata=csv_data

  def rms(self,a):
    a=list(a)
    a=[i**2 for i in a]
    return(self.np.sqrt(self.np.mean(a)))

  def func(self, tstdata):
    np=self.np
  #Initializing the blank matrix for features selection using the test dataset
    Xmn_mn = []
    Xmn_std = []
    Xmn_var = []
    Xmn_min = []
    Xmn_max = []
    Xmn_Xrms = []
    Xmn_Yrms = []
    Xmn_Zrms = []
    #for i in range(0,len(tstdata),60):
    value = tstdata
    value = np.array(value)
    #calculating the mean value of an array
    X_mn = np.mean(value,axis = 0)
    #calculating the standard deviation value of an array
    X_std = np.std(value,axis = 0)
    #calculating the variance value of an array
    X_var = np.var(value,axis = 0)
    #calculating the minimum value of an array
    X_min = np.min(value,axis=0)
    #calculating the maximum value of an array
    X_max = np.max(value,axis=0)
    # calculating the Root Mean Square value of an array
    x1,x2,x3 = value.T
    X_Xmn = np.mean(x1**2)
    X_Ymn = np.mean(x2**2)
    X_Zmn = np.mean(x3**2)
    X_Xrms = np.sqrt(X_Xmn)
    X_Yrms = np.sqrt(X_Ymn)
    X_Zrms = np.sqrt(X_Zmn)
    #use list.appen() to append a value to the list for all calculated features.
    Xmn_mn.append(X_mn)
    Xmn_std.append(X_std)
    Xmn_var.append(X_var)
    Xmn_min.append(X_min)
    Xmn_max.append(X_max)
    Xmn_Xrms.append(X_Xrms)
    Xmn_Yrms.append(X_Yrms)
    Xmn_Zrms.append(X_Zrms)
    #Calculating the 3-D feature values in 
    #Calculating 3-axis mean
    X_3mn = np.mean(Xmn_mn,axis=1)[0]
    #Calculating 3-axis minimum
    X_3min = np.min(Xmn_mn,axis=1)[0]
    #Calculating 3-axis maximum
    X_3max = np.max(Xmn_mn,axis=1)[0]
    #Calculating 3-axis standard deviation
    X_3std = np.std(Xmn_mn,axis=1)[0]
    #Calculating 3-axis variance
    X_3var = np.var(Xmn_mn,axis=1)[0]
    #Calculating 3-axis median
    X_3med = np.median(Xmn_mn,axis=1)[0]
    #Calculating 3-axis range
    X_3range = X_3max - X_3min
    #COnverting the data variables into an array individually
    # Xmn_std = np.array(Xmn_std)
    # Xmn_mn = np.array(Xmn_mn)
    # Xmn_var = np.array(Xmn_var)
    # Xmn_min = np.array(Xmn_min)
    # Xmn_max = np.array(Xmn_max)
    # Xmn_Xrms = np.array(Xmn_Xrms)
    # Xmn_Yrms = np.array(Xmn_Yrms)
    # Xmn_Zrms = np.array(Xmn_Zrms)
    # X_3max = np.array(X_3max)
    # X_3min = np.array(X_3min)
    # X_3mn = np.array(X_3mn)
    # X_3std = np.array(X_3std)
    # X_3var = np.array(X_3var)
    # X_3med = np.array(X_3med)
    # X_3range = np.array(X_3range)
    X_mean=self.st.mean(self.tstdata["X"])
    Y_mean=self.st.mean(self.tstdata["Y"])
    Z_mean=self.st.mean(self.tstdata["Z"])
    X_min=min(self.tstdata["X"])
    Y_min=min(self.tstdata["Y"])
    Z_min =min(self.tstdata["Z"])
    X_max=max(self.tstdata["X"])
    Y_max=max(self.tstdata["Y"])
    Z_max =max(self.tstdata["Z"])
    X_std=self.st.stdev(self.tstdata["X"])
    Y_std=self.st.stdev(self.tstdata["Y"])
    Z_std =self.st.stdev(self.tstdata["Z"])
    X_var=self.st.variance(self.tstdata["X"])
    Y_var=self.st.variance(self.tstdata["Y"])
    Z_var =self.st.variance(self.tstdata["Z"])
    X_rms=self.rms(self.tstdata["X"])
    Y_rms=self.rms(self.tstdata["Y"])
    Z_rms =self.rms(self.tstdata["Z"])
    self.X_tot = [X_mean, Y_mean, Z_mean, X_min, Y_min, Z_min, X_max, Y_max, Z_max, X_std, Y_std, 
                  Z_std, X_var, Y_var, Z_var, X_rms, Y_rms, Z_rms, X_3mn, X_3max, X_3min, X_3std,
                  X_3med, X_3range]
    # X_tot is the total selected features from the calculation
  def predict(self):
    self.func(self.tstdata)  
    X_tot = self.X_tot
    return(self.model.predict(X_tot)[0])
#############################################################################################################################
class Anxiety:
  import pandas as pd
  import numpy as np
  import time
  from datetime import datetime, date, timedelta
  import xgboost
  import pickle

  #Function to convert timestamp to date format
  def to_utc(self, x):
      return(self.datetime.date(x))

  #Function to convert timestamp to datetime format
  def to_utc2(self, x):
      return(self.datetime.utcfromtimestamp(x))

  #Function to convert timedelta into total no. of seconds
  def to_sec(self, x):
      a_timedelta = datetime.timedelta(x)
      timedelta_seconds = a_timedelta.total_seconds()
      return(self.timedelta_seconds)

  def __init__(self, df_call, df_sms, df_gps, df_activity, df_conversation, model_path):

    self.df_call=df_call
    self.df_sms=df_sms
    self.df_gps=df_gps
    self.df_activity=df_activity
    self.df_conversation=df_conversation

    file=open(model_path,"rb")
    self.model=self.pickle.load(file)
    file.close()


  def calls(self):
    df=self.df_call
    df['timestamp'] = df['timestamp'].apply(self.to_utc2)
    df['timestamp'] = df['timestamp'].apply(self.to_utc)
    
    #Calculating total no. of calls
    df_calls = df.groupby(['timestamp']).size().reset_index(name='count')
    
    return(df_calls['count'][0]) 

  def sms(self):
    df_sms=self.df_sms
    df_sms = df_sms[['timestamp']]
    df_sms.timestamp = df_sms.timestamp.apply(self.to_utc2)
    df_sms.timestamp = df_sms.timestamp.apply(self.to_utc)
    
    #Calculating total no. of messages
    df_sms = df_sms.groupby(['timestamp']).size().reset_index(name='count')
    
    return(df_sms['count'][0])
    

    
  def entropy(self):
    df_entropy = self.df_gps
    df_entropy.reset_index(inplace=True)
    
    #Renaming the columns to subset the mismatch in importing the file
    df_entropy.rename(columns={'index':'timestamp', 'accuracy':'lat', 'latitude':'long', 'speed':'state'}, inplace=True)
    
    df_entropy = df_entropy[['timestamp', 'lat', 'long', 'state']]
    df_entropy.timestamp = df_entropy.timestamp.apply(self.to_utc2)
    df_entropy.timestamp = df_entropy.timestamp.apply(self.to_utc)
    df_entropy.dropna(inplace=True)
    df_entropy.drop(df_entropy[df_entropy['state'] == 'moving'].index, inplace=True)
    df_entropy.drop(columns='state', inplace=True)
    
    # Merging common GPS data
    df_e = df_entropy.groupby(['timestamp', 'lat', 'long']).size().reset_index(name='count')
    
    # Calculating Total Location Entropy
    df_e.drop(columns = ['count', 'lat', 'long'], inplace=True)
    df_test = df_e.groupby(['timestamp']).size().reset_index(name='count')
    
    return(df_test['count'][0])


  def activity(self):
    df_activity=self.df_activity
    df_activity.timestamp = df_activity.timestamp.apply(self.to_utc2)
    df_activity.timestamp = df_activity.timestamp.apply(self.to_utc)
    df_act = df_activity.groupby(['timestamp', ' activity inference']).size().reset_index(name='count')
    df_act.drop(df_act[df_act[' activity inference'] != 0].index, inplace=True)
    df_act.drop(columns=[' activity inference'], inplace=True)
    df_act['minutes'] = df_act['count']*3/60
    df_act.drop(['count'], axis=1, inplace=True)
    
    return(df_act['minutes'][0])

  def conversation(self):
    df_conversation=self.df_conversation
    df_conversation['start_timestamp'] = df_conversation['start_timestamp'].apply(self.to_utc2)
    df_conversation[' end_timestamp'] = df_conversation[' end_timestamp'].apply(self.to_utc2)
    
    #calculating total conversation time per call (in seconds)
    df_conversation['diff'] = df_conversation[' end_timestamp']-df_conversation['start_timestamp']
    df_conversation['diff'] = df_conversation['diff'] / self.np.timedelta64(1, 's')
    
    df_conversation.start_timestamp = df_conversation.start_timestamp.apply(self.to_utc)
    df_conversation.drop(' end_timestamp', axis=1, inplace=True)
    
    #calculating total conversation time
    df_convers = df_conversation.groupby(['start_timestamp']).sum().reset_index()
        
    return(df_convers['diff'][0])

  def predict(self):
    call=self.calls()
    sms=self.sms()
    entropy=self.entropy()
    activity=self.activity()
    conversation=self.conversation()

    order=[call,sms,entropy,activity,conversation]
  
    return(self.model.predict([order])[0])
##############################################################################################################################  