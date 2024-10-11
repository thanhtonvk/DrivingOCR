from modules.aligner.align_card import AlignCard
from modules.detector.detector_field import DetectorField
from modules.reader.text_recognition import TextRecognition
import numpy as np
class CardRecognition:
    alignCard = AlignCard()
    detectorField = DetectorField()
    textRecognition = TextRecognition()
    def predict(self,image:np.ndarray):
        imageAligned = self.alignCard.predict(image)
        if imageAligned is None:
            return None, 'Can not detect corner'
        fieldsDetected = self.detectorField.predict(imageAligned)
        if len(fieldsDetected.keys())==0:
            return None, 'Can not detect field'
        results = {}
        for key in fieldsDetected.keys():
            images = fieldsDetected[key]
            ocr_text = ''
            for image in images:
                text = self.textRecognition.predict(image)
                ocr_text+=text
            results[key] = ocr_text
        return results



