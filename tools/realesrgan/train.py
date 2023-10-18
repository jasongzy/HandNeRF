import os.path as osp

from utils import train_pipeline

if __name__ == "__main__":
    root_path = osp.abspath(osp.dirname(__file__))
    train_pipeline(root_path)
