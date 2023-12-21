from ultralytics import YOLO
import cv2
from PIL import Image
import copy
from datetime import datetime
import os

def cv2_to_pil(cv2_image):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou

# Filtrar detecciones superpuestas usando IoU
def filter_boxes_with_iou(boxes, confidences, threshold=0.0):
    indices = list(range(len(boxes)))
    remaining_indices = []
    removed_indices = set()

    for i in range(len(indices)):
        if i in removed_indices:
            continue  # Si ya se eliminó esta detección, pasa a la siguiente

        keep = True
        for j in range(i + 1, len(indices)):
            if j in removed_indices:
                continue  # Si ya se eliminó esta detección, pasa a la siguiente

            if calculate_iou(boxes[i], boxes[j]) > threshold:
                if confidences[i] < confidences[j]:
                    removed_indices.add(i)
                    keep = False
                    break
                else:
                    removed_indices.add(j)

        if keep:
            remaining_indices.append(i)

    return remaining_indices


if __name__ == '__main__':   
    
    nombre_imagen = "4"
    model_tapa = YOLO('vino_n_v3.pt')
    cap =cv2.imread( nombre_imagen + '.jpg')

    frame = cap
    resolutions = []  
    boxes = []
    confidences = []
    if True:
        #hacemos un predict y dependiendo de ciertos parametros podemos ajustar confianza para que muestre ciertos objetos o clases
        start_time = datetime.now()

        conf = 0.4
        frame_copy =  copy.deepcopy(frame)
        tapas = model_tapa.predict(frame,show_labels=False, save_txt=True,show_conf=False,conf=conf, save=False,show=False,imgsz=1600,save_crop=True)
        cantidad_tapas = 0        
        end_time = datetime.now()
        print('Tiempo total de prediccion imagen:', end_time - start_time)
        start_time = datetime.now()
        for tapa in tapas[0].boxes:
            if( tapa.conf[0] > conf):
                cantidad_tapas = cantidad_tapas + 1
                x1, y1, x2, y2 = tapa.xyxy[0]

                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = x2 - x1
                height = y2 - y1
                resolution = width * height
                resolutions.append(resolution)
           
        if resolutions:
            average_resolution = sum(resolutions) / len(resolutions)
            print("Resolucion promedio de tapas:", average_resolution)

 

        tapas_filtradas = 0
        for tapa in tapas[0].boxes:
            if tapa.conf[0] > conf:
                x1, y1, x2, y2 = tapa.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = x2 - x1
                height = y2 - y1
                resolution = width * height

                if resolution > average_resolution/2:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(tapa.conf[0]))

                    tapas_filtradas= tapas_filtradas +1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw in green color

        end_time = datetime.now()
        print('Tiempo total filtro promedios:', end_time - start_time)

        start_time = datetime.now()
        indices = filter_boxes_with_iou(boxes, confidences, threshold=0)

        tapas_filtradas_nms= len(indices)
        for i in indices:
            box = boxes[i]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2),  (147, 20, 255), 2)
        

        end_time = datetime.now()
        print('Tiempo total filtro IOU:', end_time - start_time)

        new_folder_path = 'C:/Users/admin/Desktop/python/YoloVinoTrainingModel/resultados/'  # Provide the path to the new folder
        os.makedirs(new_folder_path, exist_ok=True)


        print('Cantidad de tapas reconocidas por modelo:'+ str(cantidad_tapas))
        print('Cantidad de tapas filtradas:'+ str(tapas_filtradas))
        print('Cantidad de tapas filtradas por nms :'+ str(tapas_filtradas_nms))

        pil_frame_copy = cv2_to_pil(frame_copy)
        pil_frame_copy.save(os.path.join(new_folder_path, nombre_imagen + "_" + str(tapas_filtradas_nms) + "_iou.jpg"))

        pil_frame = cv2_to_pil(frame)
        pil_frame.save(os.path.join(new_folder_path, nombre_imagen + "_" + str(tapas_filtradas) + "_filtrado.jpg"))

        pil_tapas_plot = cv2_to_pil(tapas[0].plot(labels=False, conf=False))
        pil_tapas_plot.save(os.path.join(new_folder_path, nombre_imagen + "_" + str(cantidad_tapas) + "_original.jpg"))

        # cv2.imshow('Boxes con iou', frame_copy)
        # cv2.imshow('Boxes frame filtrado', frame)
        # cv2.imshow('Boxes original', tapas[0].plot(labels=False, conf=False))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



