from bs4 import BeautifulSoup
import os
import random
from lxml import etree
from pprint import pprint
import prettierfier

#Get the current path
currentpath = os.getcwd()

#Get the files in the datapath
onlyfiles = [f for f in os.listdir(currentpath + '/Data') if os.path.isfile(os.path.join(currentpath + '/Data', f))]

refsoup = BeautifulSoup(features='xml')
refsoup.append(refsoup.new_tag("benchmark"))
benchmarktag = refsoup.benchmark
new_tag = refsoup.new_tag("entries")
benchmarktag.append(new_tag)
entriestag = refsoup.benchmark.entries



for file in onlyfiles:
    if file.endswith('.xml'):
        with open(os.path.join(currentpath + '/Data', file), encoding='utf-8') as fp:
            soup = BeautifulSoup(fp, 'lxml')
        allentries = soup.find('entries').find_all('entry')
        random.shuffle(allentries)
        try:
            sampled_allentries = random.sample(allentries, 5)
        except ValueError:
            sampled_allentries = random.sample(allentries, len(allentries))


        for entry in sampled_allentries:
            entriestag.append(entry)


refsname = currentpath + '/Refs.xml'

savestate = prettierfier.prettify_xml(str(refsoup))

with open(refsname, 'wb') as f:
    f.write(bytes(savestate, 'UTF-8'))
print(refsname + ' has been written')


candsoup = BeautifulSoup(features='xml')
candsoup.append(candsoup.new_tag("benchmark"))
candbenchmarktag = candsoup.benchmark
candnewtag = candsoup.new_tag("entries")
candbenchmarktag.append(candnewtag)
candentriestag = candsoup.benchmark.entries

with open(refsname, encoding='utf-8') as fp:
    soup = BeautifulSoup(fp, 'lxml')
allentries = soup.find('entries').find_all('entry')
for entry in allentries:
    newentry = candsoup.new_tag('entry', category=entry['category'], eid=entry['eid'])
    modset = entry.find('modifiedtripleset')
    modset.name = 'generatedtripleset'
    children = modset.findChildren("mtriple", recursive=False)
    for child in children:
        child.name = 'gtriple'
    newentry.append(modset)
    candentriestag.append(newentry)

candsname = currentpath + '/Cands.xml'

candsavestate = prettierfier.prettify_xml(str(candsoup))

with open(candsname, 'wb') as f:
    f.write(bytes(candsavestate, 'UTF-8'))
print(candsname + ' has been written')
