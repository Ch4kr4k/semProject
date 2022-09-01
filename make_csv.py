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
	print(len(col))
	data = pd.DataFrame(columns=col)
	data.to_csv(file, index=False)
	
create_csv()
