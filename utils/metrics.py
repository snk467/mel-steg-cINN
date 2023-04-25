import torch.nn
import wandb

import utils.logger

mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()


def accuracy(output, target):
    assert target.shape == output.shape
    return 1.0 - torch.sum(torch.sign(target) == torch.sign(output)) / target.numel()


class Metrics:
    def __init__(self, metrics_functions: dict):
        self.logger = utils.logger.get_logger(__name__)
        self.metrics_functions = metrics_functions

    def gather_batch_metrics(self, outputs, targets):
        metrics = dict(self.metrics_functions)
        loss = None
        for func in self.metrics_functions:
            if func == "Loss":
                loss = self.metrics_functions[func](outputs, targets)
                metrics[func] = loss.item()
            else:
                metrics[func] = self.metrics_functions[func](outputs, targets).item()

        return loss, metrics

    @staticmethod
    def add_metrics(metrics1, metrics2):
        if metrics1 is None:
            return dict(metrics2)

        metrics_sum = dict(metrics1)
        for metrics in metrics2:
            metrics_sum[metrics] += metrics2[metrics]

        return metrics_sum

    @staticmethod
    def divide_metrics(metrics, divisor):

        if metrics is None:
            return None

        metrics_divided = dict(metrics)

        for metrics_name in metrics:
            metrics_divided[metrics_name] /= divisor

        return metrics_divided

    def log_metrics(self, metrics, phase, step, batch_id=None):

        if batch_id is not None:
            logger_message = f"        batch {batch_id + 1}"
        else:
            logger_message = f"{phase} METRICS"

        for metrics_name in metrics:
            # Log to stdout
            logger_message += f" {metrics_name}: {metrics[metrics_name]}"

            # Log to Weights & Biases
            wandb.log({f'{phase.replace(" ", "_")}_{metrics_name}': metrics[metrics_name]}, step=step)

        # Uncomment if you want to log memory usage
        # logger_message += f" Memory: {torch.cuda.memory_allocated(device)}/{torch.cuda.get_device_properties(device).total_memory}"

        self.logger.info(logger_message)
