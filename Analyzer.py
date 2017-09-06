import ast, glob, os, sys, math
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from riotwatcher import RiotWatcher
import numpy as np
from sklearn.externals import joblib

#TeamId 100 = Blue 200 = Purple

#API_KEY = 
def wait(riotwatcher):
	while(not riotwatcher.can_make_request()):
		time.sleep(.5)
def standardDeviation(match, team, avg, index):
	summation = 0

	for player in match:
		if (player != 100 and player != 200 and player != -1 and team == player[2]):
			try:
				if(player[3 + index] == ''):
					summation += 1
				else:
					summation += math.pow((player[3 + index] - avg) , 2)
			except Exception as err:
				summation += 1
	summation = summation / float(5)
	return math.sqrt(summation)

def analyzeMatch(match, reformatted, winners):
		reformattedMatch = []
		avgA = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		avgB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		maxValA = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		maxValB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		maxInt = sys.maxint
		minValA = [maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt]
		minValB = [maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt, maxInt]
		for player in match:
			try:
				if(player != 100 and player != 200 and player != -1):
					test = player[35]
			except:
				return

		for loc in range(len(match)):
			player = match[loc]
			try:
				if (player[2] == 100):
					for index in range(0 , 35):
						try:
							if (player[index + 3] != ""):
								avgA[index] += player[index + 3]
								if (maxValA[index] < player[index + 3]):
									maxValA[index] = player[index + 3]
								if (minValA[index] > player[index + 3]):
									minValA[index] = player[index + 3]
						except:
							pass
				else:
					for index in range(0 , 35):
						try:
							if (player[index + 3] != ""):
								avgB[index] += player[index + 3]
								if (maxValB[index] < player[index + 3]):
									maxValB[index] = player[index + 3]
								if (minValB[index] > player[index + 3]):
									minValB[index] = player[index + 3]
						except:
							pass
			except:
				pass
		for index in range(0, 35):
			avgA[index] = avgA[index] / float(5)
			avgB[index] = avgB[index] / float(5)
			reformattedMatch.append(avgA[index] - avgB[index])
			reformattedMatch.append(maxValA[index] - maxValB[index])
			reformattedMatch.append(minValA[index] - minValB[index])

			reformattedMatch.append(standardDeviation(match, 100, avgA[index], index))
			reformattedMatch.append(standardDeviation(match, 200, avgB[index], index))
		reformattedMatch.extend(avgA)
		reformattedMatch.extend(map(lambda x: x*x, avgA))
		reformattedMatch.extend(map(lambda x: np.sin(x), avgA))
		reformattedMatch.extend(avgB)
		reformattedMatch.extend(map(lambda x: x*x, avgB))
		reformattedMatch.extend(map(lambda x: np.sin(x), avgB))
		reformattedMatch.extend(maxValA)
		reformattedMatch.extend(map(lambda x: x*x, maxValA))
		reformattedMatch.extend(map(lambda x: np.sin(x), maxValA))
		reformattedMatch.extend(maxValB)
		reformattedMatch.extend(map(lambda x: x*x, maxValB))
		reformattedMatch.extend(map(lambda x: np.sin(x), maxValB))
		reformattedMatch.extend(minValA)
		reformattedMatch.extend(map(lambda x: x*x, minValA))
		reformattedMatch.extend(map(lambda x: np.sin(x), minValA))
		reformattedMatch.extend(minValB)
		reformattedMatch.extend(map(lambda x: x*x, minValB))
		reformattedMatch.extend(map(lambda x: np.sin(x), minValB))
		reformatted.append(reformattedMatch)
		winners.append(match[len(match) - 1])

def reformat(matches):
	reformatted = []
	winners = []
	for match in matches:
		if (match != None and len(match) != 0):
			if (match[len(match) - 1] != 100 and match[len(match) - 1] != 200 and match[len(match) - 1] != -1):
				for realMatch in match:
					analyzeMatch(realMatch, reformatted, winners)
			else:
				analyzeMatch(match, reformatted, winners)	
	return [reformatted, winners]

def predictMatch():
	os.chdir("./")
	matches = []
	for file in glob.glob("*.res"):
		with open(file, "r") as inputFile:
			for line in inputFile:
				matches.append(ast.literal_eval(line[:-1]))
	result = reformat(matches)
	reformatted = result[0]
	winners = result[1]

	clf = RandomForestClassifier(n_estimators=1000)
	clf = clf.fit(reformatted, winners)
	print("Found Tree")
	print("\n")
	current = []
	with open("currentGame.txt", "r") as inputFile:
		for line in inputFile:
				current.append(ast.literal_eval(line[:-1]))
	current.append(current[0])
	currentResult = reformat(current)
	current = currentResult[0]
	currentWinners = currentResult[1]
	print("Prediction: " + str(clf.predict(current)[0]))
	print("Probability: " + str(clf.predict_proba(current)[0]))

# predictMatch()

os.chdir("./")
matches = []
for file in glob.glob("*.res"):
	with open(file, "r") as inputFile:
		for line in inputFile:
			matches.append(ast.literal_eval(line[:-1]))
result = reformat(matches)
reformatted = result[0]
winners = result[1]
print(str(len(reformatted)) + " training games")
est = 1200
clf = RandomForestClassifier(n_estimators=est)
print("Number of estimators: " + str(est))
clf = clf.fit(reformatted, winners)
print("Found Tree")
print("\n")
tester = []
with open("testData.txt", "r") as inputFile:
	for line in inputFile:
			tester.append(ast.literal_eval(line[:-1]))
testerResult = reformat(tester)
tester = testerResult[0]
testerWinners = testerResult[1]

dex = 0
prediction = clf.predict(tester)
total = 0
correct = 0
threshold = 0.6
for probablility in clf.predict_proba(tester):
	isRight = prediction[dex] == testerWinners[dex]
	dex += 1
	for ind in probablility:
		if (ind >= threshold):
			total += 1
			if(isRight):
				correct += 1
print("Percent Correct with " + str(threshold) + " threshold: " + str(correct / float(total)))
print("Num games correct: " + str(correct))
print("Num games incorrect: " + str(total - correct))
print("Total games within threshold: " + str(total))
print("Total games: " + str(dex + 1))
score = clf.score(tester, testerWinners)
print("Score: " + str(score))
other = joblib.load("tree/randomForest.pkl")
otherScore = other.score(tester, testerWinners)
print("Other Score: " + str(otherScore))
if (score > otherScore):
	joblib.dump(clf, "tree/randomForest.pkl")

print("\nCross Validation Fitting")
slf = RandomForestClassifier(n_estimators=est)
reformatted.extend(tester)
winners.extend(testerWinners)
acc = cross_validation.cross_val_score(slf, reformatted, winners)
print("Avg Acc: " + str(sum(acc) / len(acc)))


# baseNames = ["URwins", "URtotalChampionKills", "URtotalTurretsKilled", "URtotalMinionKills", "URtotalNeutralMinionsKilled", "URtotalAssists", 
# "Rwins", "RtotalChampionKills", "RtotalTurretsKilled", "RtotalMinionKills", "RtotalNeutralMinionsKilled", "RtotalAssists",
# "RCtotalPhysicalDamageDealt", "RCtotalTurretsKilled", "RCtotalSessionsPlayed", "RCtotalAssists", "RCtotalDamageDealt", 
# "RCmostChampionKillsPerSession", "RCtotalPentaKills", "RCmostSpellsCast", "RCtotalDoubleKills", "RCmaxChampionsKilled", 
# "RCtotalDeathsPerSession", "RCtotalSessionsWon", "RCtotalGoldEarned", "RCtotalTripleKills", "RCtotalChampionKills", 
# "RCmaxNumDeaths", "RCtotalMinionKills", "RCtotalMagicDamageDealt", "RCtotalQuadraKills", "RCtotalUnrealKills", 
# "RCtotalDamageTaken", "RCtotalSessionsLost", "RCtotalFirstBlood"]
# avgAmB = []
# for name in baseNames:
# 	avgAmB.append(str(name) + "_avgDifference")
# avgA = []
# for name in baseNames:
# 	avgA.append(str(name) + "_avgA")
# avgB = []
# for name in baseNames:
# 	avgB.append(str(name) + "_avgB")
# maxAmB = []
# for name in baseNames:
# 	maxAmB.append(str(name) + "_maxDifference")
# maxA = []
# for name in baseNames:
# 	maxA.append(str(name) + "_maxA")
# maxB = []
# for name in baseNames:
# 	maxB.append(str(name) + "_maxB")
# minAmB = []
# for name in baseNames:
# 	minAmB.append(str(name) + "_minDifference")
# minA = []
# for name in baseNames:
# 	minA.append(str(name) + "_minA")
# minB = []
# for name in baseNames:
# 	minB.append(str(name) + "_minB")
# sdA = []
# for name in baseNames:
# 	sdA.append(str(name) + "_standardDeviationA")
# sdB = []
# for name in baseNames:
# 	sdB.append(str(name) + "_standardDeviationB")
# keys = []
# for index in range(0, 35):
# 	keys.append(avgAmB[index])
# 	keys.append(maxAmB[index])
# 	keys.append(minAmB[index])
# 	keys.append(sdA[index])
# 	keys.append(sdB[index])
# keys.extend(avgA)
# keys.extend(avgB)
# keys.extend(maxA)
# keys.extend(maxB)
# keys.extend(minA)
# keys.extend(minB)

# dic = {}
# importances = clf.feature_importances_
# for index in range(0, len(keys)):
# 	dic[str(keys[index])] = importances[index]
# sortedDic = sorted(dic, key=dic.get, reverse=True)
# summation = 0
# for key, value in dic.items():
# 	summation += value
# output = open("StatsByImportance.txt", "w")
# for element in sortedDic:
# 	output.write(str(element) + ": "+ str(100 * (dic[str(element)] / float(summation))))
# 	output.write("\n")
# output.close()
# simpleDic = {}
# for element in baseNames:
# 	simpleDic[element] = dic[str(element) + "_avgDifference"]
# 	simpleDic[element] += dic[str(element) + "_avgA"]
# 	simpleDic[element] += dic[str(element) + "_avgB"]
# 	simpleDic[element] += dic[str(element) + "_maxA"]
# 	simpleDic[element] += dic[str(element) + "_maxB"]
# 	simpleDic[element] += dic[str(element) + "_maxDifference"]
# 	simpleDic[element] += dic[str(element) + "_minA"]
# 	simpleDic[element] += dic[str(element) + "_minB"]
# 	simpleDic[element] += dic[str(element) + "_minDifference"]
# 	simpleDic[element] += dic[str(element) + "_standardDeviationA"]
# 	simpleDic[element] += dic[str(element) + "_standardDeviationB"]
# 	simpleDic[element] = (simpleDic[element] / float(summation)) * 100
# simpleSortedDic = sorted(simpleDic, key=simpleDic.get, reverse=True)
# output = open("SimpleStatsByImportance.txt", "w")
# for element in simpleSortedDic:
# 	output.write(str(element) + ": " + str(simpleDic[str(element)]))
# 	output.write("\n")
# output.close()






