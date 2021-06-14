import requests
from requests.auth import HTTPProxyAuth
import os
file_path = os.path.dirname(os.path.abspath(__file__))

from requests.utils import DEFAULT_CA_BUNDLE_PATH
print(DEFAULT_CA_BUNDLE_PATH)

proxies = {
     'https': 'http://gateway.zscalertwo.net:443'
}

r = requests.get(url='https://xkcd.com', proxies=proxies)
print(r.status_code)
print(r.text)
