import torch
import numpy


class EarlyStopping:
    def __init__(self, args: dict):
        """
        :param args:
        """
        self.patience = args["early_stopping_patience"]
        self.verbose = args["verbose"]
        self.path = args["corpus_path"] + "Models/" + args['job_id']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = numpy.Inf

    def __call__(self, val_loss, models):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            i = 0
            for model in models:
                i += 1
                self.save_checkpoint(val_loss, model, i)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            i = 0
            for model in models:
                i += 1
                self.save_checkpoint(val_loss, model, i)
            self.counter = 0

    def save_checkpoint(self, val_loss, model: torch.nn.Module, name: str) -> None:
        """
        :param val_loss:
        :param model:
        :param name:
        :return: Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path + "/checkpoint" + str(name) + ".pt")
        self.val_loss_min = val_loss
