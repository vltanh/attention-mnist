from torch.utils.tensorboard import SummaryWriter

class TensorboardHelper():
    def __init__(self, path):
        assert path != None, "path is None"
        self.writer = SummaryWriter(log_dir=path)
    
    def update_loss(self, phase, value, step):
        self.writer.add_scalar(tag=phase, scalar_value=value, global_step=step)

    def update_metric(self, phase, metric, value, step):
        self.writer.add_scalar('{}/{}'.format(phase, metric), value, step)
    
    