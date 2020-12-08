# Use pip3 install -I xgboost==0.90
import warnings
import os
import traceback

warnings.filterwarnings("ignore")
############################### For optimising warnings and Tracebacks. #####################################

############################### Importing Necessary Packages for the Program. ################################
try:
    from gevent.pywsgi import WSGIServer
except ImportError:
    os.system('sudo python3 -m pip install gevent')
    from gevent.pywsgi import WSGIServer

try:
        import requests
except ImportError:
        os.system("sudo python3 -m pip install requests")
        import requests
        
from flask import Flask,request,jsonify,render_template
from flask_json import FlaskJSON,JsonError, json_response, as_json
import datetime
import json

from colorama import init
from termcolor import colored
from pygments import highlight, lexers, formatters
### Importing custom classes. ####################################################

from mental_health.models import *
from mental_health.sensors import *
####################################### Important Functions and objects. ###############################################
map_={"depression":["gps"],
         "bioplar":["activity"]}

def prediction(model,data):

    if(model=="depression"):
        predictor=Depression(csv_data=data["gps"],model_path="models/depression.sav")
        return(predictor.predict())
    if(model=="bipolar"):
        predictor=Bipolar(csv_data=data["activity"],model_path="models/bipolar.sav")
        return(predictor.predict())
    if(model=="Sleep_hours"):
        predictor=Sleep_hours(dark_data=data["dark"], activity_data=data["activity"], gps_data=data["gps"], phonelock_data=data["phonelock"],audio_inference=data["audio"],model_file="models/Sleep_mean.sav")        
        return(predictor.predict())
    if(model=="Sleep_sqi"):
        predictor=Sleep_sqi(dark_data=data["dark"], activity_data=data["activity"], gps_data=data["gps"], phonelock_data=data["phonelock"],audio_inference=data["audio"],model_file="models/Sleep_sqi.sav")        
        return(predictor.predict())
    if(model=="Stress"):
        predictor=Stress(csv_data=data["activity"],model_path="models/Stress.sav")
        return(predictor.predict())
    if(model=="Anxiety"):
        predictor=Anxiety(df_call=data["call_log"], df_sms=data["sms"], df_gps=data["gps"], df_activity=data["activity"], df_conversation=data["conversation"], model_path="models/Anxiety.sav")
        return(predictor.predict())

####################################### Main Program. #######################################################

app = Flask(__name__)

FlaskJSON(app)
### Response Test ################################################################
@app.route('/test')
def home():
	return("Success!")

### Model Input ##################################################################
@app.route('/sensor',methods=['POST'])
def recieve():
    ask=request.get_json(force=True)
    input_paths=ask["input"]
    #validate(ask["model"],input_types,map_)
    data=get_input(input_paths).data_files
    ask["prediction"]=str(prediction(ask["model"],data))
    return(ask)





if __name__ == '__main__':
    init()
    address='0.0.0.0'
    port=5619
    http_server = WSGIServer((address, port), app)
    #os.system('clear')
    print("Server started Successfully!")
    print("Listening on port: "+colored(str(port),"green"))
    print("Go to "+colored("/test","yellow")+" to test the response")
    print("Press CTRL+C to terminate the server")
    print("\n")
    http_server.serve_forever()    
