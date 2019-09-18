import sys
import os

SPAM_CODE = 0
HAM_CODE = 1
START_POINT = 1
NUMBER_OF_FOLDERS = 3
emailDataSet = []

FILE_NAME = "enron"


def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def file_get_contents(filename):
    print(fileName)
    with open(filename,"r") as f:
        return f.read()


trainingSet = []
counter = 0
for i in range(START_POINT, NUMBER_OF_FOLDERS+1):
    currentDir = sys.argv[1] + "/" + FILE_NAME
    hamFileList = get_files(currentDir + str(i) + "/ham")
    spamFileList = get_files(currentDir + str(i) + "/spam")
    for spamFile in spamFileList:
        try:
            fileName = currentDir + str(i) + "/spam/" + spamFile
            trainingSet.append({"emailContent":file_get_contents(fileName),"class":SPAM_CODE})
        except UnicodeDecodeError:
            counter+=1
    for hamFile in hamFileList:
        try:
            fileName = currentDir + str(i) + "/ham/" + hamFile
            trainingSet.append({"emailContent":file_get_contents(fileName),"class":HAM_CODE})
        except UnicodeDecodeError:
            counter += 1

print(len(trainingSet))
