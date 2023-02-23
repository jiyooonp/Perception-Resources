'''
Delete files that are not annotated
'''
import os
import glob

fileName = 'set3'
path = '../dataset/'+fileName
os.chdir(path)

def removeExtension(fileName):
    return fileName.split('.')[0]

filesInDir_jpeg = glob.glob("*.jpeg")
filesInDir_xml = glob.glob("*.xml")


filesInDir_jpeg.sort()
filesInDir_xml.sort()

filesInDir_jpeg = list(map(removeExtension, filesInDir_jpeg))
filesInDir_xml = list(map(removeExtension, filesInDir_xml))

jpegPointer = 0
for xmlPointer in range(len(filesInDir_xml)):
    print("working on:", filesInDir_xml[xmlPointer], 'pointer at:', filesInDir_jpeg[jpegPointer])
    if filesInDir_xml[xmlPointer] != filesInDir_jpeg[jpegPointer]:
        while filesInDir_xml[xmlPointer]>filesInDir_jpeg[jpegPointer]:
            os.remove(filesInDir_jpeg[jpegPointer]+".jpeg")
            jpegPointer+=1
            print("removed:", filesInDir_jpeg[jpegPointer])
    jpegPointer += 1
while jpegPointer<len(filesInDir_jpeg):
    os.remove(filesInDir_jpeg[jpegPointer] + ".jpeg")
    jpegPointer+=1