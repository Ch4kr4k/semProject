import pandas as pd

import mediapipe as mp

mp_hands = mp.solutions.hands


#For creating csv file
def create_csv():
	col =[]
	for i in range(21):
		col.append(str(mp_hands.HandLandmark(i).name))
	print(len(col))
	data = pd.DataFrame(columns=col)
	data.to_csv('data.csv', index=False)
	
create_csv()
