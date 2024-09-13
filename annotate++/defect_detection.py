import os
from mmdet.apis import inference_detector
from mmcv import Config
from mmdet.models import build_detector
import torch
import numpy as np
from PIL import Image

class DefectDetection:
    def __init__(self, model_path=None, config_file=None, device="cpu", score_thresholds=[]):
        self.device = device
        self.score_thresholds = score_thresholds
        self.cfg = Config.fromfile(config_file)
        self._model = self.create_model_mmdet(model_path)
        self.scores=[]
        if torch.cuda.is_available():
            self.device = "cuda"
            self._model.cuda()
        self._classes = self.cfg.CLASSES
        if len(self.score_thresholds) == 0:
            self.score_thresholds = [0.3 for x in self._classes]

    def create_model_mmdet(self, model_path):
        model = build_detector(
            self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg
        )
        model.CLASSES = self.cfg.CLASSES

        model.cfg = self.cfg
        model = self.load_weights_mmdet(model, model_path, self.device)
        return model

    def load_weights_mmdet(self, model, checkpoint_path, device):
        epoch_model = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(epoch_model["state_dict"])
        del epoch_model
        return model.eval()

    def show_inference(self, image):
        """
        Works for both image_paths and numpy image arrays
        """
        self.scores = []
        boxes = inference_detector(self._model, image)
        result = self._model.show_result(image, boxes, thickness=10, score_thr=0.01)
        result = {
            "image": Image.fromarray(result),
            "result_boxes": self.get_preds_mmdet(boxes),
            # "scores": self.get_preds_mmdet(boxes)[1],
        }
        return result

    def get_preds_mmdet(self, boxes):
        result = {}
        scores = []
        for i, arr in enumerate(boxes):
            if len(arr) > 0:
                for box in arr:
                    if box[-1] > self.score_thresholds[i]:
                        scores += [round(box[-1], 2)]
                        self.scores = list(scores)
                        result[tuple(box[:4])] = self._classes[i]
        return result
