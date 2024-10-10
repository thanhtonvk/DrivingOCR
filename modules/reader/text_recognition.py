from modules.reader import ocr
import numpy as np
import cv2
class TextRecognition:

    model = ocr.Reader()
    def predict(self,image:np.ndarray):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = self.model.readtext(image)
        if len(result)>0:
            text_result = ' '.join([i[1] for i in result if i[2]>0.1])
            return text_result
        return ''