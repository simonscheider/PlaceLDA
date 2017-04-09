#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      simon
#
# Created:     09/04/2017
# Copyright:   (c) simon 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from lxml import html
import requests
import urllib2

from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


url = "http://speeltuinbankaplein.nl/"
#url = 'http://www.obsooginal.nl/johandewitt/'
page = requests.get(url, headers=headers)



#print page.content
tree = html.fromstring(page.content)

soup = BeautifulSoup(page.content)

print soup.title.string#soup.prettify()

##divs = soup.find_all("div")
##ps = soup.find_all("p")
##strongs =soup.find_all("strong")
##bs=soup.find_all("b")
##for d in divs:
##    if d.string !=None:
##      print d.bstring
##
##for p in ps:
##    if p.string !=None:
##        print p.string
##
##for b in bs:
##    if b.string !=None:
##        print b.string
##
##for strong in strongs:
##    if strong.string !=None:
##        print strong.string


# kill all script and style elements and all links
for script in soup(["script", "style", "a"]):
    script.extract()    # rip it out

text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

print(text)

#print soup.get_text()


##buyers = tree.xpath('//div[@title="buyer-name"]/text()')
###This will create a list of prices
##prices = tree.xpath('//span[@class="item-price"]/text()')
##
##print 'Buyers: ', buyers
##print 'Prices: ', prices
