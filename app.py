import gradio as gr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 金門具有代表性的栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺五種物種。我們來挑戰五種類別總共用五十張照片, 看能不能打造一個神經網路學會辨識這五種類別。
# 讀入栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺資料圖檔
image_folders = ['Merops_philippinus', 'pavo_cristatus', 'Upupa_epops', 'King_Crab', 'otter']

# 為了後面的需要，我們將五種類別照片的答案用 `labels` 呈現
labels = ["栗喉蜂虎", "藍孔雀", "戴勝", "鱟", "歐亞水獺"]

num_classes = len(labels)

base_dir = './classify_image/'

# 載入並檢視訓練完成的模型。
model = load_model('my_cnn_model.h5') # Loading the Tensorflow Saved Model (PB)
print(model.summary())

# 注意現在主函數做辨識只有五個種類。而且是使用我們自行訓練的 model!
def classify_image(inp):
    inp = inp.reshape((-1, 256, 256, 3))
    inp = preprocess_input(inp)
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(num_classes)}

image = gr.Image(shape=(256, 256), label="栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺照片")
label = gr.Label(num_top_classes=num_classes, label="AI ResNet50V2遷移式學習辨識結果")
some_text="我能辨識金門栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺。找張金門栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺照片來考我吧!"

# 我們將金門栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺數據庫中的圖片拿出來當作範例圖片讓使用者使用
sample_images = []
for i in range(num_classes):
    thedir = base_dir + image_folders[i]
    for file in os.listdir(thedir):
        if file == ".git" or file == ".ipynb_checkpoints":
          continue
        sample_images.append(base_dir + image_folders[i] + '/' + file)

# 最後，將所有東西組裝在一起，就大功告成了！
iface = gr.Interface(fn=classify_image,
             inputs=image,
             outputs=label,
             title="AI 栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺辨識機",
             description=some_text,
             examples=sample_images, live=True)

iface.launch()
