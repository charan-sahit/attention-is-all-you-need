class NoamSchedule:
    """
    lrate = dmodel^0.5*min(step^0.5, step*warmup_steps^-1.5)
    """
    def __init__(self, d_model, warmup_steps) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def lr(self, step):
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(
            step**-0.5,
            step*(self.warmup_steps**-1.5)
        )