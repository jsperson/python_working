import requests
import os
file_path = os.path.dirname(os.path.abspath(__file__))

proxies = {
    "http": "gateway.zscalertwo.net:10077"
}

r = requests.get('https://xkcd.com/353/', proxies=proxies)  
#print(r.status_code)
#print(dir(r))
#print(help(r))

#Download HTML
#print(r.text)

#Get an image
r = requests.get('https://imgs.xkcd.com/comics/python.png')
#print(r.content)
#save as a file (wb stands for write binary)
#with open(file_path + '/comic.png','wb') as f:
#    f.write(r.content)

print(r.status_code)

#r.ok will return True for anything below 400
print(r.ok)

#headers
print(r.headers)

#use httpbin.org
payload = {'page': 2, 'count': 25}
r = requests.get('https://httpbin.org/get', params=payload)
print(r.text)
print(r.url)

payload = {'username':'corey', 'password':'testing'}
r = requests.post('https://httpbin.org/post', data=payload)
#print(r.text)
print(r.json())
r_dict = r.json()
print(r_dict['form'])

#authentication
r = requests.get('https://httpbin.org/basic-auth/corey/testing', auth=('corey','testing'))
print(r.text)
print(r) # note wrong username/password in auth will respond with 401

#timeout - delay in URL for 6, but set timeout to 3 and you'll see a timeout
r = requests.get('https://httpbin.org/delay/6', timeout = 3)
print(r)
