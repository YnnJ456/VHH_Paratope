import os
import re
import sys
import pandas as pd


class LoadDataset:
    def __init__(self):
        pass

    def readVHH_Fasta(self, File):
        if not os.path.exists(File):
            print('Error: "' + File + '" does not exist.')
            sys.exit(1)
        seqNameList = []
        seq_gapList = []
        seq_noGapList = []
        with open(File, 'r') as f:
            for line in f.readlines():
                if '>' in line:
                    startLocList = [index for index, char in enumerate(line) if char == '[']
                    endLocList = [index for index, char in enumerate(line) if char == ']']
                    seqName = line[startLocList[0]+1:endLocList[0]]
                    seq = line[startLocList[1]+1:endLocList[1]]
                    times = line[startLocList[2]+1:endLocList[2]]
                    seqNameList.append(f'{seqName}_{seq}_{times}')
                else:
                    seq_gap = re.sub('[^ARNDCQEGHILKMFPSTWYV0-]', '', ''.join(line).upper())
                    seq_noGap = re.sub('[^ARNDCQEGHILKMFPSTWYV0]', '', ''.join(line).upper())
                    seq_gapList.append(seq_gap)
                    seq_noGapList.append(seq_noGap)
        seqDict = dict(zip(seqNameList, seq_noGapList))
        seqMinusDict = dict(zip(seqNameList, seq_gapList))

        return seqDict, seqMinusDict
