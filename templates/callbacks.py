from tensorflow.keras.callbacks import Callback


class MyCallbacks(Callback):
    """
    override `on_epoch_end` for stop_training threshold
    """
    def __init__(self, target_acc: float):
        super(MyCallbacks, self).__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        try:
            acc = logs.get('acc')
        except KeyError:
            acc = logs.get('accuracy')
        if acc >= self.target_acc:
            print(f"\nStop training since accuracy is over target_acc(>={self.target_acc})")
            self.model.stop_training = True
        return
