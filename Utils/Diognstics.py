# hook that will catptuer the percentage of dead ReLUs in each layer
def dead_percentage_hook(name, dead_percent):
    def hook(module, input, output):
        with torch.no_grad():
            percentage = (output == 0).float().mean().item() *100
            dead_percent[name] = percentage
    return hook
