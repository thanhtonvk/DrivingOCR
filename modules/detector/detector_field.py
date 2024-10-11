from ultralytics import YOLO
import numpy as np
class DetectorField:
    model = YOLO('models/field_detect_yolov11.pt')
    classes = model.names
    def predict(self,image:np.ndarray):
        result = self.model.predict(image,verbose = False)[0]
        cls = result.boxes.cls.cpu().detach().numpy().astype('int')
        xyxy = result.boxes.xyxy.cpu().detach().numpy().astype('int')
        result_dict = {}
        for i in set(cls):
            result_dict[self.classes[i]] = []
        for id_class, box in zip(cls,xyxy):
            xmin,ymin,xmax,ymax = box
            cropped = image[ymin:ymax,xmin:xmax]
            result_dict[self.classes[id_class]].append(cropped)
        return result_dict
