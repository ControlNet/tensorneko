from torch.nn import LeakyReLU, GELU, ELU, ReLU, CrossEntropyLoss, L1Loss, MSELoss, BCELoss


class StringGetter:
    activation_mapping = {
        "LEAKYRELU": LeakyReLU,
        "GELU": GELU,
        "ELU": ELU,
        "RELU": ReLU
    }
    loss_mapping = {
        "CROSSENTROPYLOSS": CrossEntropyLoss,
        "L1LOSS": L1Loss,
        "MSELOSS": MSELoss,
        "BCELOSS": BCELoss
    }

    mapping = {
        "activation": activation_mapping,
        "loss": loss_mapping
    }

    def __init__(self, type: str):
        self.type = type
        self.mapping = StringGetter.mapping[self.type]

    def __call__(self, name: str):
        return self.get(name)

    def get(self, name: str):
        return self.mapping[name.upper()]


activation_getter = StringGetter("activation")
loss_getter = StringGetter("loss")
