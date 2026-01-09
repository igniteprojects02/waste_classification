import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO
from collections import deque, Counter
import serial
import time

# --- CONFIGURATION ---
YOLO_THRESHOLD = 0.50
TM_THRESHOLD = 0.60 
BUFFER_SIZE = 15
REQUIRED_VOTES = 10 

# --- SERIAL CONFIGURATION (ADJUSTED FOR ESP32) ---
SERIAL_PORT = 'COM7'  # Your Port
BAUD_RATE = 115200    # <--- CHANGED TO MATCH ESP32 "Serial.begin(115200)"

# --- SETUP SERIAL ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"✅ Serial Connected on {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"❌ Serial Error: {e}")
    ser = None

# --- SETUP MODELS ---
TM_MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

YOLO_WASTE_MAP = {
    46: "BIODEGRADABLE", 47: "BIODEGRADABLE", 49: "BIODEGRADABLE", 
    50: "BIODEGRADABLE", 51: "BIODEGRADABLE", 52: "BIODEGRADABLE", 
    53: "BIODEGRADABLE", 54: "BIODEGRADABLE", 55: "BIODEGRADABLE", 
    39: "NON-BIODEGRADABLE", 40: "NON-BIODEGRADABLE", 41: "NON-BIODEGRADABLE", 
    42: "NON-BIODEGRADABLE", 43: "NON-BIODEGRADABLE", 44: "NON-BIODEGRADABLE", 
    45: "NON-BIODEGRADABLE", 67: "NON-BIODEGRADABLE", 77: "NON-BIODEGRADABLE"
}

print("Loading Models...")
tm_model = load_model(TM_MODEL_PATH, compile=False)
yolo_model = YOLO('yolov8n.pt') 
tm_labels = open(LABELS_PATH, "r").readlines()

camera = cv2.VideoCapture(0)
cv2.namedWindow("Smart Bin Debug", cv2.WINDOW_NORMAL)

# --- STATE VARIABLES ---
result_buffer = deque(maxlen=BUFFER_SIZE) 

print("-" * 50)
print("STARTING DETECTION... Press 'ESC' to exit.")
print("-" * 50)

while True:
    ret, image = camera.read()
    if not ret: break

    # Variables for logging
    log_source = "NONE"
    log_conf = 0.0
    instant_result = "WAITING" 
    
    # 1. YOLO CHECK
    results = yolo_model(image, verbose=False, conf=YOLO_THRESHOLD)
    yolo_detected = False
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in YOLO_WASTE_MAP:
                instant_result = YOLO_WASTE_MAP[cls_id]
                yolo_detected = True
                log_source = f"YOLO ({yolo_model.names[cls_id]})"
                log_conf = float(box.conf[0])
                
                # Draw Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if instant_result == "BIODEGRADABLE" else (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                break 

    # 2. TM CHECK
    if not yolo_detected:
        img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img_input = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        img_input = (img_input / 127.5) - 1

        tm_pred = tm_model.predict(img_input, verbose=0)
        tm_index = np.argmax(tm_pred)
        tm_conf = tm_pred[0][tm_index]

        # Log what TM sees regardless of threshold for debugging
        log_source = f"TM (Idx {tm_index})"
        log_conf = tm_conf

        if tm_index != 0 and tm_conf > TM_THRESHOLD:
            if tm_index == 1:
                instant_result = "BIODEGRADABLE"
            elif tm_index == 2:
                instant_result = "NON-BIODEGRADABLE"
        else:
            instant_result = "WAITING"

    # 3. VOTING LOGIC
    result_buffer.append(instant_result)
    votes = Counter(result_buffer)
    winner, count = votes.most_common(1)[0]

    final_decision = "WAITING" 
    serial_signal = '0' 

    if winner != "WAITING" and count >= REQUIRED_VOTES:
        final_decision = winner
        if final_decision == "BIODEGRADABLE":
            serial_signal = '1'
        elif final_decision == "NON-BIODEGRADABLE":
            serial_signal = '2'

    # 4. SEND TO SERIAL
    if ser and ser.is_open:
        try:
            ser.write(serial_signal.encode('utf-8'))
        except Exception as e:
            print(f"Serial Error: {e}")

    # 5. PRINT TO TERMINAL (DEBUG LOG)
    # Format: [Source | Conf] -> [Instant Result] -> [Votes] -> [Final Output]
    
    # Shorten string for cleaner print
    short_res = instant_result[:4] # WAIT, BIOD, NON-
    short_dec = final_decision[:4]
    
    log_string = (
        f"SRC: {log_source:<15} | "
        f"CNF: {log_conf:.2f} | "
        f"INST: {short_res:<4} | "
        f"VOTES: {winner[:3]}({count}/{BUFFER_SIZE}) | "
        f"SIG: {serial_signal}"
    )
    print(log_string)

    # 6. DISPLAY ON SCREEN
    if final_decision == "BIODEGRADABLE":
        disp_color = (0, 255, 0)
    elif final_decision == "NON-BIODEGRADABLE":
        disp_color = (0, 0, 255)
    else:
        disp_color = (200, 200, 200)

    # Visual Vote Bar
    vote_strength = int((count / BUFFER_SIZE) * image.shape[1])
    cv2.rectangle(image, (0, 0), (vote_strength, 15), disp_color, -1) 

    # Text Status
    cv2.rectangle(image, (0, 15), (image.shape[1], 80), (0, 0, 0), -1)
    status_text = f"SIG: {serial_signal} | {final_decision}"
    cv2.putText(image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, disp_color, 2, cv2.LINE_AA)
    
    cv2.imshow("Smart Bin Debug", image)
    if cv2.waitKey(1) == 27: break

camera.release()
if ser: ser.close()
cv2.destroyAllWindows()