#5
#Hosts a web interface with Flask and SocketIO.

#Captures webcam feed and processes it with MediaPipe.

#Uses the trained Random Forest model to classify ASL signs.

#Applies delayed character detection (6 seconds default) and auto-space logic (after 10 seconds of no hand).

#Streams the processed video feed and recognized sentence to the frontend in real time.

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import time

# Suppress MediaPipe deprecation warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Load the ML model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

# Global state
global_sentence = ""

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('reset')
def handle_reset():
    global global_sentence
    print("Reset requested by client.")
    global_sentence = ""
    socketio.emit('prediction', {'text': ''})  # Clear on frontend too

def generate_frames():
    global global_sentence

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Sorry',
        27: 'Please'
    }

    last_detection_time = time.time()
    space_timer = time.time()
    DETECTION_DELAY = 6
    SPACE_DELAY = 10

    while True:
        data_aux = []
        x_, y_ = [], []

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_detected = False
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detected = True

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = max(prediction_proba[0])
                    predicted_character = labels_dict[int(prediction[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence*100:.2f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                (0, 0, 0), 3, cv2.LINE_AA)
                except Exception:
                    predicted_character = ""

        if hand_detected and predicted_character and time.time() - last_detection_time >= DETECTION_DELAY:
            global_sentence += predicted_character
            last_detection_time = time.time()
            space_timer = time.time()

        if not hand_detected:
            countdown_time = SPACE_DELAY - int(time.time() - space_timer)
            if countdown_time <= 0:
                global_sentence += " "
                space_timer = time.time()

        # Show sentence on screen
        cv2.putText(frame, f"Sentence: {global_sentence}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if not hand_detected:
            cv2.putText(frame, f"Space in: {countdown_time}s", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Send to frontend
        socketio.emit('prediction', {'text': global_sentence})

        # Return frame to browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
