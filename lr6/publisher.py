import paho.mqtt.client as mqtt
import pandas as pd
import json
import time

broker = "localhost"
port = 1883
topic = "parkinson/data_with_label"

client = mqtt.Client()
client.connect(broker, port, 60)
client.loop_start()

df = pd.read_csv("lr6/parkinsons_data.csv")
# Отправляем только X и status (как метку)
for _, row in df.iterrows():
    features = row.drop(labels=["name", "status"]).to_list()
    label = int(row["status"])
    payload = {"features": features, "label": label}
    client.publish(topic, json.dumps(payload), qos=1)
    print("Published:", payload)
    time.sleep(0.5)

client.loop_stop()
client.disconnect()
