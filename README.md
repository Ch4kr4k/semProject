# Final Year Project

## installing dependencies

> installing python libraries

    pip3 install -r requirements.txt

## Usage for create_csv.py

```
python3 create_csv.py filename
```

File name should be without .csv also if no filename is not specified then we use the default name.
The create_csv.py python file create a csv by appending all the Landmarks of the hand into the list, then it convertsto a dataframe which is then it converts to a csv.

## Usage for get_data.py

    python3 get_data.py filename

File name should be without .csv also if no filename is not specified then we use the default name.
This Python File use opencv and mediapipe to capture the Landmarks of the hand and panda to create a csv of those Landmarks


## usage of dino.py

        sudo python3 dino.py

> just a python program to control chrome://dino game with control

![dino](./presentation/dino.gif)