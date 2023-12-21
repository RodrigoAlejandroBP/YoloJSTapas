from ultralytics import YOLO
import cv2

#eso es lo que estamos generando con el nuevo dataset, el error es pequeño y se podría evitar haciendo un calculo del promedio del tamaño de los detectados  y borrando los que sean inferior a eso... daría bien exacto, se puede mejorar aun el dataset 100%  pero con eso yo creo que basta 
if __name__ == '__main__':   
         
    yolo = YOLO('yolov8n.pt',)
    yolo.train(data='data.yaml', epochs=110, imgsz=1600, batch=2)