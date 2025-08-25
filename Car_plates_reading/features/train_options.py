class TrainOptions:
    def __init__(self):
        self.workers = 2
        self.img_size = 640
        self.batch = 16
        self.epochs = 50

def train_options():
    return TrainOptions()