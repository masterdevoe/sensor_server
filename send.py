import requests
import json
import sys
from pygments import highlight, lexers, formatters

# ask={"model":"depression", 
#      "input":{
#                "gps":"data/depression/test_modsevere.csv",
#             }
# }

#ask={"model":"Sleep_sqi",
#     "input":{
#               "dark":"data/Sleep_hours/dark_u03.csv",
#               "activity":"data/Sleep_hours/activity_u03.csv",
#               "gps":"data/Sleep_hours/gps_u03.csv",
#               "phonelock":"data/Sleep_hours/phonelock_u03.csv",
#               "audio":"data/Sleep_hours/audio_u03.csv"
#     }
#
#}




# ask={"model":"bipolar", 
#      "input":{
#                "activity":"data/bipolar/condition_12.csv",
#             }
# }

#ls=[]
#for i in range(20):
#    ask={"model":"Stress",
#         "input":{
#                   "activity":"data/Stress/iter_{}.csv".format(i+1),
#                }

#    }
    

#    response=requests.post("http://0.0.0.0:5619/sensor",json=ask)

#    response=response.json()
#    ls.append(int(response["prediction"]))


#    formatted_json = json.dumps(response, indent=4)
#    json_ = highlight(str(formatted_json), lexers.JsonLexer(), formatters.TerminalFormatter())
#    print(json_)

#pred="Predictions: "+str(ls)
#pred=pred.replace(",","->").replace("[","").replace("]","")
#print(pred)

ask={"model":"Anxiety", 
      "input":{
                "activity":"data/Anxiety/activity_u01.csv",
                "sms":"data/Anxiety/sms_u01.csv",
                "conversation":"data/Anxiety/conversation_u01.csv",
                "call_log":"data/Anxiety/call_log_u01.csv",
                "gps":"data/Anxiety/gps_u01.csv"
             }
 }


response=requests.post("http://0.0.0.0:5619/sensor",json=ask)

response=response.json()
   

#print(response)

formatted_json = json.dumps(response, indent=4)
colorful_json = highlight(str(formatted_json), lexers.JsonLexer(), formatters.TerminalFormatter())
print(colorful_json)