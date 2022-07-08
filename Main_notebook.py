# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Расспознование изображений рака на МРТ снимках головного мозга
#
# Искомая задача состоит в следующем: пусть есть единичное изображение МРТ мозга, программа должна сказать есть ли на этом фото рак. Более формально на каждое фото программа должна выдать вероятность присутствия рака. Это задача относится к классу задач, называемых задачи бинарной классификации. 
#
# В машинном обучении для решения задач бинарной классификации используются различные методы, такие например как деревья решений или логистическая регрессия. Так как наши входные данные это изображения это сильно сужает возможность произвольного выбора метода. То есть, лучше всего будет использовать методы наиболее хорошо работающие с изображениями, такие например как свёрточные нейронные сети.
#
# Для начала нужно получить данные загрузить их и сделать предобработку.
#
# Предварительно загрузили датасет с сайта kaggle и поместили его в папку Data нашего проекта.
#
# https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

# %%
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import imutils
import torch 
import torchvision
import PIL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


sns.set(rc={'figure.figsize':(8, 8)})

# %% [markdown]
# Для начала можно можно загрузить единичное изображение и посмотреть как это выглядит.

# %%
A = cv2.imread('Data/yes/Y1.jpg')

# %%
sns.heatmap(A[:,:,0], cmap='bone')

# %% [markdown]
# Так как данные мы загрузили локально на диск и их колличество их небольшое мы можем смеловсе их сразу загрузить в оперативную память для работы с ними

# %%
df = []
labels = []

# %% [markdown]
# Следующие циклы загружают все фотографии из папки. Помимо этого некоторые фотографии имеют три канала цветов, несмотря на то что они монохромные. В таких фотографиях мы оставляем только один произвольные канал, какой именно - не важно так как мотографии монохромные и все три канала одинаковы.

# %%
for path in os.listdir("Data/yes"):

    temp = cv2.imread('Data/yes/'+path)
    if len(temp.shape) == 3:
        temp = temp[:,:,0]
    df.append(temp)
    labels.append(1)

for path in os.listdir("Data/no"):

    temp = cv2.imread('Data/no/'+path)
    
    if len(temp.shape) == 3:
        temp = temp[:,:,0]
    df.append(temp)
    labels.append(0)



# %% [markdown]
# функция для отрисовки большого колличества изображений:

# %%
def show_image(df, n, k):
    plt.figure(figsize=(n*10, k*10))
    for i in range(n*k):
        plt.subplot(k, n, i+1)
        plt.axis("off")
        plt.imshow(df[i], cmap="bone")
    plt.show()


# %%
show_image(df, 10, 3)


# %% [markdown]
# Тут сразу видно две проблеммы, которые могут помешать алгоритму выдать хороший результат. Во первых фотографии все разного размера, во вторый некоторые изображения имеют большую чёрную часть по краям. Небходимо обрезать все фотографии до краёв изображения мозга. Делать это будем используя библиотеки OpenCV и imutils. Подробнее про алгоритом можно прочитать в следующей статье:

# %% [markdown]
# https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
# ![Изображение шагов алгоритма поиска краёв контура изображения](alg_steps.png)

# %%
def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.GaussianBlur(img, (5, 5), 0)
        
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return set_new


# %%
df = crop_imgs(df)

# %%
show_image(df, 10, 3)

# %% [markdown]
# Видно что стало лучше, изображения больше похожи друг на друга.
#
# Далее необходимо прибегнуть к особой технике, а именно Data Augmentation. Это способ увеличить размер тренирующей выборки при помощи небольшого видоизменения исходных данных. Необходимо это из за того, что наш датасет очень маленький. Это кстати типичная проблема если решается задача из области медицины.
#
# Да изображений такими изменениями может быть зеркалирование, поворот, сдвиг, обрезка, изменение цвета, аффинное преобразование и т.д.
#
# Мы будем использовать библиотеку PyTorch для нейросетей, она же предоставляет функционал для Data Augmentation.
#
# Более подробно об этом можно прочитать в следующей статье:

# %% [markdown]
# https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py

# %%
transforms = torchvision.transforms.Compose([
    # Это композиция преобразований, изображение проходит через каждое по порядку
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((230,230),interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop((224, 224)), # Обрезка озображения
    torchvision.transforms.RandomHorizontalFlip(0.5), # зеркалирование изображения по горизонтали с вероятностью 0.5
                                                      # Особенно актуально для симметричного мозга
    torchvision.transforms.RandomAffine(7, (0.03,0.03), (0.93, 1), (-3,3,-3,3), 
                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    # Случайное Афинное преобразование: 
    # Повернуть в случайную сторону не более чем на 7 градусов
    # сместить по вертикали и горизонтали не более чем на 3 %
    # Уменьшить на не более чем в 0.93 раз
    # Сдвинуть изображение не более чем на 3 по каждой из осей
    #torchvision.transforms.Normalize(0,1)
])

# %% [markdown]
# Посмотрим какие преобразования получаются:

# %%
n = 8
k = 5

plt.figure(figsize=(n*10, k*10))
for i in range(n * k):
    plt.subplot(k, n, i+1)
    plt.axis("off")
    plt.imshow(transforms(df[i//n]).numpy()[0,:,:], cmap="bone")
    
plt.show()

# %% [markdown]
# Так как изображения всё равно очень похожи друг на друга, необходимо не допустить того, что бы разные варианты одного изображения оказались одновременно и в тренировочной и тестовой выборке. Поэтому для начала разделим датасет на тренировочную и тестовую выборку.

# %%
type(df[0])

# %%
train_dsX, test_dsX, train_dsy, test_dsy = train_test_split(df, labels, stratify=labels, test_size=0.20)

# %% [markdown]
# Теперь мы можем увеличит наш датасет, колличество копий исходных изображений я взял 5. Это немного, так сделанно потому что получаемые изображения всё равно сильно похожи друга на друга.

# %%
k = 5  #Сколько копий одного изображения будем делать
train_X = []
train_y = []

for i in range(len(train_dsX)):
    for j in range(k):
        train_X.append(transforms(train_dsX[i]).numpy()[0,:,:].tolist())
        train_y.append(train_dsy[i])
train_X = np.array(train_X)
train_y = np.array(train_y)

test_X = []
test_y = []

for i in range(len(test_dsX)):
    for j in range(k):
        test_X.append(transforms(test_dsX[i]).numpy()[0,:,:].tolist())
        test_y.append(test_dsy[i])
test_X = np.array(test_X)
test_y = np.array(test_y)

# %% [markdown]
# Теперь необходимо провести нормализацию для того что бы алгоритмы лучше работали.
# Так как torchvision.transforms.Normalize заставить работать не удалось, будем использовать sklearn
# и его StandardScaler

# %%
scalar = StandardScaler()
scalar.fit(np.vstack((train_X, test_X)).reshape(-1, 224**2))
train_X = scalar.transform(train_X.reshape(-1, 224**2))
test_X = scalar.transform(test_X.reshape(-1, 224**2))

train_X = train_X.reshape(-1, 224, 224)
test_X = test_X.reshape(-1, 224, 224)

# %%
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# %%
show_image(train_X, 8, 5)

# %%
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

# %%
model_ft

# %%
