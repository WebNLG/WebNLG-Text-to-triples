# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from bert_score import score
from bs4 import BeautifulSoup
import os
import regex as re
import itertools
import pickle
import statistics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
import sys

currentpath = os.getcwd()

def getRefs(filepath):
    with open(filepath, encoding='utf-8') as fp:
        refssoup = BeautifulSoup(fp, 'lxml')

    refsentries = refssoup.find('benchmark').find('entries').find_all('entry')

    allreftriples = []

    for entry in refsentries:
        entryreftriples = []
        modtriplesref = entry.find('modifiedtripleset').find_all('mtriple')
        for modtriple in modtriplesref:
            entryreftriples.append(modtriple.text)
        allreftriples.append(entryreftriples)

    newreflist = []

    for entry in allreftriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allreftriples, newreflist

def getCands(filepath):
    with open(filepath, encoding='utf-8') as fp:
        candssoup = BeautifulSoup(fp, 'lxml')

    candssentries = candssoup.find('benchmark').find('entries').find_all('entry')

    allcandtriples = []

    for entry in candssentries:
        entrycandtriples = []
        modtriplescand = entry.find('generatedtripleset').find_all('gtriple')
        for modtriple in modtriplescand:
            entrycandtriples.append(modtriple.text)
        allcandtriples.append(entrycandtriples)

    newcandlist = []

    for entry in allcandtriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcandtriples, newcandlist

def calculateAllBertScores(newreflist, newcandlist):
    allprecision = []
    allrecall = []
    allf1 = []

    for idx, candidate in enumerate(newcandlist):
        candidateprecision = []
        candidaterecall = []
        candidatef1 = []
        for triple in candidate:
            tripleprecision = []
            triplerecall = []
            triplef1 = []
            for reference in newreflist[idx]:
                P, R, F1 = score([triple], [reference], lang='en', rescale_with_baseline=True)
                precision = list(P.numpy())[0]
                recall = list(R.numpy())[0]
                newf1 = list(F1.numpy())[0]
                print(triple, reference)
                print(P, R, F1)
                print(precision, recall, newf1)
                print('------------------------------------------')
                tripleprecision.append(precision)
                triplerecall.append(recall)
                triplef1.append(newf1)
            candidateprecision.append(tripleprecision)
            candidaterecall.append(triplerecall)
            candidatef1.append(triplef1)
        allprecision.append(candidateprecision)
        allrecall.append(candidaterecall)
        allf1.append(candidatef1)

    '''
    with open(currentpath + '/Data/Precisionscores.pkl', 'wb') as f:
        pickle.dump(allprecision, f)

    with open(currentpath + '/Data/Recallscores.pkl', 'wb') as f:
        pickle.dump(allrecall, f)

    with open(currentpath + '/Data/F1scores.pkl', 'wb') as f:
        pickle.dump(allf1, f)
    '''
    return allprecision, allrecall, allf1

def calculateSystemScore(allprecision, allrecall, allf1, newreflist, newcandlist):
    selectedprecision = []
    selectedrecall = []
    selectedf1 = []
    alldicts = []

    # Get all the permutations of the number of scores given per candidate, so if there's 4 candidates, but 3 references, this part ensures that one of
    # The four will not be scored
    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) > len(newreflist[idx]):
            # Get all permutations
            choosecands = list(itertools.permutations([x[0] for x in enumerate(allf1[idx])], len(allf1[idx][0])))
            # The permutations in different orders are not necessary: we only need one order without the number of candidates we're looking at
            choosecands = set([tuple(sorted(i)) for i in choosecands])  # Sort inner list and then use set
            choosecands = list(map(list, choosecands))  # Converting back to list
        else:
            # Otherwise, we're just going to score all candidates
            choosecands = [list(range(len(newcandlist[idx])))]

        # Get all permutations in which the scores can be combined
        if len(newcandlist[idx]) > len(newreflist[idx]):
            choosescore = list(itertools.permutations([x[0] for x in enumerate(allf1[idx][0])], len(newreflist[idx])))
            choosescore = [list(x) for x in choosescore]
        else:
            choosescore = list(itertools.permutations([x[0] for x in enumerate(allf1[idx][0])], len(newcandlist[idx])))
            choosescore = [list(x) for x in choosescore]

        # Get all possible combinations between the candidates and the scores
        combilist = list(itertools.product(choosecands, choosescore))

        totaldict = {'totalscore': 0}

        for combination in combilist:
            combiscore = 0
            # Take the combination between the candidate and the score
            zipcombi = list(zip(combination[0], combination[1]))
            collectedf1scores = []
            collectedprecisionscores = []
            collectedrecallscores = []
            for zc in zipcombi:
                f1score = allf1[idx][zc[0]][zc[1]]
                combiscore += f1score
                collectedf1scores.append(f1score)

                precisionscore = allprecision[idx][zc[0]][zc[1]]
                recallscores = allrecall[idx][zc[0]][zc[1]]
                collectedprecisionscores.append(precisionscore)
                collectedrecallscores.append(recallscores)

            # If the combination is the highest score thus far, or the first score, make it the totaldict
            if (combiscore > totaldict['totalscore']) or (len(totaldict) == 1):
                totaldict = {'totalscore': combiscore, 'combination': combination, 'f1scorelist': collectedf1scores,
                             'precisionscorelist': collectedprecisionscores, 'recallscorelist': collectedrecallscores}

        differencebetween = abs(len(newcandlist[idx]) - len(newreflist[idx]))
        differencelist = [0] * differencebetween
        totaldict['f1scorelist'] = totaldict['f1scorelist'] + differencelist
        totaldict['precisionscorelist'] = totaldict['precisionscorelist'] + differencelist
        totaldict['recallscorelist'] = totaldict['recallscorelist'] + differencelist
        selectedprecision = selectedprecision + totaldict['precisionscorelist']
        selectedrecall = selectedrecall + totaldict['recallscorelist']
        selectedf1 = selectedf1 + totaldict['f1scorelist']
        alldicts.append(totaldict)

    return statistics.mean(selectedprecision), statistics.mean(selectedrecall), statistics.mean(selectedf1), alldicts

def calculateExactTripleScore(reflist, candlist):
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]
    #First get all the classes by combining the triples in the candidatelist and referencelist
    allclasses = newcandlist + newreflist
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))

    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(newcandlist)
    mrbin = lb.fit_transform(newreflist)

    precision = precision_score(mrbin, mcbin, average='macro')
    recall = recall_score(mrbin, mcbin, average='macro')
    f1 = f1_score(mrbin, mcbin, average='macro')
    return precision, recall, f1

def attributeScores(reflist, candlist, alldicts):
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]

    allrefattributes = []
    allcandattributes = []

    for idx, totaldict in enumerate(alldicts):
        bestcombi = list(zip(totaldict['combination'][0], totaldict['combination'][1]))
        linkedcands = []
        linkedrefs = []
        for triplecombi in bestcombi:
            linkedcands.append(newcandlist[idx][triplecombi[0]])
            linkedrefs.append(newreflist[idx][triplecombi[1]])
        for idx2, triple in enumerate(linkedcands):
            candattributes = linkedcands[idx2].split(' | ')
            refattributes = linkedrefs[idx2].split(' | ')
            allcandattributes = allcandattributes + candattributes
            allrefattributes = allrefattributes + refattributes

        differencebetween = abs(len(candlist[idx]) - len(reflist[idx]))
        emptycands = ['noMatch'] * (differencebetween * 3)
        emptyrefs = ['noScores'] * (differencebetween * 3)
        allcandattributes = allcandattributes + emptycands
        allrefattributes = allrefattributes + emptyrefs

    allclasses = allcandattributes + allrefattributes
    allclasses = list(set(allclasses))
    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)

    allcandattributes = [[x] for x in allcandattributes]
    allrefattributes = [[x] for x in allrefattributes]
    mcbin = lb.fit_transform(allcandattributes)
    mrbin = lb.fit_transform(allrefattributes)

    precision = precision_score(mrbin, mcbin, average='micro')
    recall = recall_score(mrbin, mcbin, average='micro')
    f1 = f1_score(mrbin, mcbin, average='micro')
    return precision, recall, f1

def main(reffile, candfile):
    reflist, newreflist = getRefs(reffile)
    candlist, newcandlist = getCands(candfile)
    bertprecision, bertrecall, bertf1 = calculateAllBertScores(newreflist, newcandlist)
    systembertprecision, systembertrecall, systembertf1, alldicts = calculateSystemScore(bertprecision, bertrecall, bertf1, newreflist, newcandlist)
    systemprecision, systemrecall, systemf1 = calculateExactTripleScore(reflist, candlist)
    attributeprecision, attributerecall, attributef1 = attributeScores(reflist, candlist, alldicts)
    print('BERT score precision: ' + str(systembertprecision))
    print('BERT score recall: ' + str(systembertrecall))
    print('BERT score F1: ' + str(systembertf1))
    print('Triple match score (full triple) precision: ' + str(systemprecision))
    print('Triple match score (full triple) recall: ' + str(systemrecall))
    print('Triple match score (full triple) F1: ' + str(systemf1))
    print('Attribute based score precision: ' + str(attributeprecision))
    print('Attribute based score recall: ' + str(attributerecall))
    print('Attribute based score F1: ' + str(attributef1))

#main(currentpath + '/Refs.xml', currentpath + '/Cands2.xml')
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])