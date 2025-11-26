import paho.mqtt.client as mqtt
from joblib import load
import numpy as np
import json
from datetime import datetime
import csv
import os

MODEL_PATH = "lr6/rf_parkinson_model.joblib"
LOG_CSV = "lr6/predictions_log.csv"
TOPIC_SUB = "parkinson/data_with_label"
TOPIC_PUB = "parkinson/result"

model = load(MODEL_PATH)

# prepare CSV header if not exists
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "features", "true_label", "prediction"])

client = mqtt.Client()


def on_connect(client, userdata, flags, rc):
    print("Connected to broker, rc =", rc)
    client.subscribe("parkinson/#")


def on_message(client, userdata, msg):
    try:
        text = msg.payload.decode()
        # try json first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "features" in data:
                features = np.array(data["features"], dtype=float).reshape(1, -1)
                true_label = int(data.get("label", -1))
            else:
                # if payload is comma-separated features (old format)
                arr = list(map(float, text.split(",")))
                features = np.array(arr, dtype=float).reshape(1, -1)
                true_label = -1
        except json.JSONDecodeError:
            # fallback: csv string
            arr = list(map(float, text.split(",")))
            features = np.array(arr, dtype=float).reshape(1, -1)
            true_label = -1

        pred = int(model.predict(features)[0])
        ts = datetime.utcnow().isoformat()
        # publish JSON result (includes true if present)
        out = {"prediction": pred, "timestamp": ts}
        if true_label != -1:
            out["true"] = true_label
        client.publish(TOPIC_PUB, json.dumps(out), qos=1)
        print("Pred:", pred, "True:", true_label)

        # append to CSV log
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, features.tolist()[0], true_label, pred])

    except Exception as e:
        print("Processing error:", e)


client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.loop_forever()
