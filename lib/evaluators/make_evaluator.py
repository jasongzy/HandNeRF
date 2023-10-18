import imp

from lib.evaluators.if_nerf import Evaluator


def _evaluator_factory(cfg) -> Evaluator:
    module = cfg.evaluator_module
    path = cfg.evaluator_path
    evaluator = imp.load_source(module, path).Evaluator()
    return evaluator


def make_evaluator(cfg):
    return None if cfg.skip_eval else _evaluator_factory(cfg)
