import urllib3
import json
http = urllib3.PoolManager()
r = http.request('GET', 'http://api.openweathermap.org/data/2.5/weather?q=faro&appid=0b3c1c23e3e51f2a7f11ff0c819ad7dd&units=metric')
data =json.loads(r.data)
print(data['main'])