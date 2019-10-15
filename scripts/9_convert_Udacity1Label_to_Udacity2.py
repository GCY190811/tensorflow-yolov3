#1. Convert Dataset1(crowdai) label format:
#  xmin,ymin,xmax,ymax,Frame,Label(word),Preview URL 
#   to Dataset2(autti) label format:
#  Frame xmin ymin xmax ymax occluded label(word)

import os

Dataset1Label = "/home/guo/moDisk/Dataset/object-detection-crowdai/labels.csv"

sections = Dataset1Label.split("/")
sections[-1] = "convert_"+sections[-1]
OutputFile = "/".join(sections)
print("Save csv: {}".format(OutputFile))
headline = True

with open(OutputFile, "w+") as wf:
    with open(Dataset1Label, "r+") as rf:
        for originLine in rf:
            s = originLine.split(",")
            line = s[4] + " " + s[0] + " " + s[1] + " " + s[2] + " " +s[3] + " " + "0" + " " + s[5].lower()
            if headline == True:
                headline = False
                print(line)
            else:
                wf.writelines(line+"\n")

            



