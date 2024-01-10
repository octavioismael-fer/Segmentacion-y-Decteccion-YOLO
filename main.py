# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")

# Realizar Video Captura
cap = cv2.VideoCapture(0)

# Creamos bucle
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Leemos los resultados
    resultados =  model.predict(frame, imgsz = 640, conf = 0.75)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos los fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar el programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()