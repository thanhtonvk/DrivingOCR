from docaligner import DocAligner
import docsaidkit as D
from docsaidkit import Backend
from docaligner import DocAligner, ModelType
import torch
import cv2
import numpy as np
gpu = 0 if torch.cuda.is_available() else -1


class AlignCard():
    model = DocAligner(
        gpu_id=gpu,  # GPU ID, set to -1 if not using GPU
        backend=Backend.cpu,  # Choose the computational backend, can be Backend.cpu or Backend.cuda
        model_type=ModelType.point  # Choose the model type, can be ModelType.heatmap or ModelType.point
    )
    def predict(self,image:np.ndarray):
        result = self.model(image)
        if result.doc_polygon:
            rflat_img = result.gen_doc_flat_img(image_size=(539, 856))
            return rflat_img
        return None




    
