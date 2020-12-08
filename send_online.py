import requests

ask={"model":"depression", 
     "input":{
               "gps":"data/test_severe.csv"
            }
}

response=requests.post("http://34.93.93.105:5619/sensor",json=ask)

print(response.json())
