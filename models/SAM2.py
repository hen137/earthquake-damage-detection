class SAM2():
    def __init__(self, train_loader, net, optimizer, val_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.net = net
        self.optimizer = optimizer

    def adjust_lr(self, curr_iter, all_iter):
        return

    def train(self):
        return

    def validate(self, curr_epoch):
        return