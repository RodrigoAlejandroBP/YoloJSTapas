from ultralytics import YOLO
import cv2

#eso es lo que estamos generando con el nuevo dataset, el error es pequeño y se podría evitar haciendo un calculo del promedio del tamaño de los detectados  y borrando los que sean inferior a eso... daría bien exacto, se puede mejorar aun el dataset 100%  pero con eso yo creo que basta 
if __name__ == '__main__':   
         
    model = YOLO('vino_n_v3.pt')  # load a custom trained model

    # Export the model
    model.export(format='onnx')