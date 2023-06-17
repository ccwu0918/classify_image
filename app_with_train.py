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

# 1. 讀入栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺資料圖檔
image_folders = ['Merops_philippinus', 'pavo_cristatus', 'Upupa_epops', 'King_Crab', 'otter']

# 為了後面的需要，我們將五種類別照片的答案用 `labels` 呈現
labels = ["栗喉蜂虎", "藍孔雀", "戴勝", "鱟", "歐亞水獺"]

num_classes = len(labels)

base_dir = './classify_image/'

thedir = base_dir + image_folders[0]
os.listdir(thedir)

data = []
target = []
for i in range(5):
    thedir = base_dir + image_folders[i]
    image_fnames = os.listdir(thedir)
    for theimage in image_fnames:
        if theimage == ".git" or theimage == ".ipynb_checkpoints":
          continue
        img_path = thedir + '/' + theimage
        img = load_img(img_path , target_size = (256,256))
        x = img_to_array(img)
        data.append(x)
        target.append(i)

# 2. 針對資料圖檔使用 ResNet 進行預處理
x_train = preprocess_input(data)

y_train = to_categorical(target, num_classes)

# 3. 用 ResNet50 打造我們的神經網路
resnet = ResNet50V2(include_top=False, pooling="avg")    

# 再來就是正式打造我們遷移學習版的函數學習機! 可以發現我們只是加入了最後
model = Sequential()
model.add(resnet)
model.add(Dense(5, activation='softmax')) # 這裡的 5 表示，輸出結果為 5 個類別

# 我們是遷移式學習, 原本 ResNet 的部份我們當然沒有重新訓練的意思。於是就設這邊不需要訓練。
resnet.trainable = False

# 欣賞我們的神經網路
model.summary()

# 組裝我們的函數學習機
# 這裡我們用分類時非常標準的 categorical_crossentropy, 順便試試有名的 adam 學習法，當然也可以試試 sgd 看效果如何。

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 我們可以發現原來有超過兩千萬個參數, 經我們借來以後, 只有 1,0245 個參數要調。

# 4. 訓練 fit
# 這裡我們全部的資料也只有 50 筆, 所以 batch_size 就選擇 25 了，同時訓練的回合數設定為 10 回合
model.fit(x_train, y_train, batch_size=25, epochs=10)

# 5. 預測
# 我們先用 model.evaluate 看一下模型表現得如何
loss, acc = model.evaluate(x_train, y_train)
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

# 6. 將訓練完成的模型儲存起來，可供日後直接略過訓練直接載入訓練完成的模型進行辨識。
model.save('my_cnn_model.h5')

# 7. 載入並檢視訓練完成的模型。
model = load_model('my_cnn_model.h5') # Loading the Tensorflow Saved Model (PB)
print(model.summary())

# 8. 用 gradio 打造栗喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺辨識 Web App!

# 注意現在主函數做辨識只有五個種類。而且是使用我們自行訓練的 model!
def classify_image(inp):
    inp = inp.reshape((-1, 256, 256, 3))
    inp = preprocess_input(inp)
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(5)}

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
             title="AI 喉蜂虎、藍孔雀、戴勝、鱟及歐亞水獺辨識機",
             description=some_text,
             examples=sample_images, live=True)

iface.launch()
