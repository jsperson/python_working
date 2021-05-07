import urllib.request, urllib.parse, urllib.error

from bs4 import BeautifulSoup

import re

url = input('Enter URL - ')

html = urllib.request.urlopen(url).read()

links = re.findall(b'href="(http://.*?)"', html)

for link in links:
    print(link.decode())