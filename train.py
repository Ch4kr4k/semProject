import tensorflow as tf
assert tf.__version__.startswith('2')
import argparse

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

class Train():

    def __init__(self):
        self.train_data =None
        self.validation_data=None
        self.test_data = None
        self.model = None

    def get_data(self,dataset_path):
        data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
        )
        self.train_data, rest_data = data.split(0.8)
        self.validation_data, self.test_data = rest_data.split(0.5)

        

    def train_model(self,model_path,epoch=20):
        hparams = gesture_recognizer.HParams(export_dir=model_path,epochs=epoch)
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        self.model = gesture_recognizer.GestureRecognizer.create(
            train_data=self.train_data,
            validation_data=self.validation_data,
            options=options
        )
        self.eval()
    
    def eval(self):
        loss, acc = self.model.evaluate(self.test_data, batch_size=1)
        print(f"Test loss:{loss}, Test accuracy:{acc}")

    def save_model(self):
        self.model.export_model()



def process(opt):
    model = Train()
    model.get_data(opt.data)
    model.train_model(opt.path)
    if opt.save:
        model.save_model()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, default="exported_model", help='path to model wight'
        )
    parser.add_argument(
        '--save', type=bool, default=False, help='Save trained weights or not'
    )
    parser.add_argument(
        '--data', type=str, default='', help='Path to dataset file'
    )
    parser.add_argument(
        '--dest',default='',help='the destination folder to saved trained weights'
    )
    parser.add_argument(
        '--epoch',type=int,default=10,help='epoch'
    )
   
    opt = parser.parse_args()
    process(opt=opt)
