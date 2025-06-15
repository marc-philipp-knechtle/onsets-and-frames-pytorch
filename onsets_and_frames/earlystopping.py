class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        print(f'\nEarly stopping patience (counter needs to reach patience): {self.counter}')
        print(f'Current score: {score}')
        print(f'Best score: {self.best_score}')
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model
            self.counter = 0