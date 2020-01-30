import requests
from bs4 import BeautifulSoup

page = "https://www.saksfifthavenue.com/Handbags/shop/_/N-52jzot"
result = requests.get(page)
print(result)
if result.status_code == 200:
    soup = BeautifulSoup(result.content, "html.parser")
    print(soup)

#table = soup.find('div',{'class':'MUQY0'})