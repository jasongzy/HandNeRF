import imp

from .interhands_renderer import Renderer as IHRenderer
from .tpose_renderer import Renderer as TRenderer


def make_renderer(cfg, network) -> TRenderer | IHRenderer:
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer = imp.load_source(module, path).Renderer(network)
    return renderer
