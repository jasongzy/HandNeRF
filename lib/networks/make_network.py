import imp

from torch.nn import Module


def make_network(cfg) -> Module:
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network
