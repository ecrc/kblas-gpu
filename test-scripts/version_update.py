import os
import re
from datetime import date

PATH=".."
VERSION_FILE="../VERSION.txt"
excpetion_list = 'flops.h'

all_files = [x[0] + "/" + y for x in os.walk(PATH) for y in os.listdir(x[0]) if re.search(r'\.[ch][hup]?[hp]?$', y) and y not in excpetion_list ]

with open(VERSION_FILE, "r") as fd:
    version = fd.readline()[:-1]

date = date.today()
strdate = str(date)

for fname in all_files:
    # newline='' keeps each line ending as it is in the file
    with open(fname, "r+", newline='') as fd:
        lines = fd.readlines()
        fd.seek(0)
        fd.truncate()
        for line in lines:
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            ind = line.find(r"@version")
            if ind != -1:
                newline = line[:ind+9]+version+eol
                if newline != line:
                    print("Warning: updated version of {}".format(fname))
                line = newline
            ind = line.find(r"@date")
            if ind != -1:
                newline = line[:ind+6]+strdate+eol
                if newline != line:
                    print("Warning: updated date of {}".format(fname))
                line = newline
            fd.write(line)
    print("File {} was succesfully processed".format(fname))
