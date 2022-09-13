import sys

import pandas as pd

import mediapipe as mp

mp_hands = mp.solutions.hands

if len(sys.argv)==1:
        file='data.csv'
        print(len(sys.argv))
else:
        file = sys.arv[1]
        file=file+'.csv'
#For creating csv file

def create_csv():
        col =[]
        for i in range(21):
                col.append(str(mp_hands.HandLandmark(i).name))
        new_cols = []
        for i in col:
                new_cols.append(i+"_X")
                new_cols.append(i+"_Y")
                new_cols.append(i+"_Z")
        data = pd.DataFrame(columns=new_cols)
        data.to_csv(file, index=False)

create_csv()
