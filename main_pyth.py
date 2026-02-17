import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import os
import cv2
import matplotlib.pyplot as plt
CONTENT_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/4. Face Recognition - Neural Style Transfer/Custom Neural Style/images/content/"

STYLE_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/4. Face Recognition - Neural Style Transfer/Custom Neural Style/images/style/"
content_image=cv2.imread(CONTENT_PATH+"louvre.jpg")
print(content_image.shape)

if content_image is None:
    print("No Image Found")

else:
    content_image=cv2.cvtColor(content_image,cv2.COLOR_BGR2RGB)
    content_image=content_image.astype("float")/255.


generated_image=content_image.copy()
noise=np.random.uniform(size=generated_image.shape,low=-0.25,high=0.25)
generated_image+=noise
generated_image=np.clip(generated_image,a_min=0.0,a_max=1.0)


plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.title("Content Image")
plt.axis("off")
plt.imshow(content_image)

plt.subplot(1,2,2)
plt.title("Initialized Generated Image")
plt.axis("off")
plt.imshow(generated_image)
base_model=VGG19(include_top=False,weights="imagenet",input_shape=(200,200,3))
base_model.trainable=False
print(len(base_model.layers))

middle_layer=base_model.layers[12]
print(middle_layer)
base_model.layers
intermediate_model=Model(inputs=base_model.input,outputs=middle_layer.output)
intermediate_model.summary()
def compute_content_cost(a_C,a_G):

    print(a_C.shape)
    
    
    m,n_h,n_w,n_c=a_G.shape

    a_C_unrolled=tf.reshape(tf.transpose(a_C,[3,1,2,0]),(n_c,-1))
    a_G_unrolled=tf.reshape(tf.transpose(a_G,[3,1,2,0]),(n_c,-1))

    J_content=tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)/(4*n_c*n_h*n_w)

    return J_content
STYLE_LAYERS={
    "block1_conv2":0.4,
    "block2_conv2":0.3,
    "block3_conv2":0.1,
    "block4_conv2":0.1,
    "block5_conv2":0.1
}


def compute_style_cost(a_S_array,a_G_array):

    m,n_h,n_w,n_c=a_G_array[0].shape
    J_style=0   

    for i,lambd in enumerate(list(STYLE_LAYERS.values())):
        a_S_unroll=tf.reshape(tf.transpose(a_S_array[i],[3,1,2,0]),(n_c,-1))
        a_G_unroll=tf.reshape(tf.transpose(a_G_array[i],[3,1,2,0]),(n_c,-1))

        a_S_style=tf.matmul(a_S_unroll,tf.transpose(a_S_unroll))
        a_G_style=tf.matmul(a_G_unroll,tf.transpose(a_G_unroll))

        J_style_layer=tf.reduce_sum((a_G_style-a_S_style)**2)/(4*(n_c*n_h*n_w)**2)

        J_style+=J_style_layer*lambd

    return J_style

@tf.function()
def total_cost(J_content,J_style,alpha=10,beta=40):
    
    J_total=alpha*J_content + beta*J_style
    return J_total
content_image=cv2.resize(content_image,(200,200))
content_image=np.expand_dims(content_image,axis=0)

generated_image=cv2.resize(generated_image,(200,200))
generated_image=np.expand_dims(generated_image,axis=0)

a_C=intermediate_model(content_image)
print("a_C shape : ",a_C.shape)


# J_content=compute_content_cost(a_C,a_G)
# print(J_content)
def get_layer_outputs(model):

    outputs=[model.get_layer(name).output for name in list(STYLE_LAYERS.keys())]
    
    temp_model=Model(inputs=[model.input],outputs=outputs)
    return temp_model

vgg_layer_outputs=get_layer_outputs(base_model)

style_image=cv2.imread(STYLE_PATH+"monet.jpg")
if style_image is None:
    print("No Image")
else:
    style_image=cv2.cvtColor(style_image,cv2.COLOR_BGR2RGB)
    style_image=cv2.resize(style_image,(200,200))
    style_image=style_image.astype("float")/255.
    style_image=np.expand_dims(style_image,axis=0)
    print(style_image.shape)

a_S_array=vgg_layer_outputs(style_image)
print(type(a_S_array),len(a_S_array))

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
generated_image=tf.Variable(generated_image,dtype=tf.float32)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:

        a_G=intermediate_model(generated_image)
        J_content=compute_content_cost(a_C,a_G)

        a_G_array=vgg_layer_outputs(generated_image)
        J_style=compute_style_cost(a_S_array,a_G_array)

        J=total_cost(J_content,J_style)

    grad=tape.gradient(J,generated_image)
    optimizer.apply_gradients([(grad,generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image,0.0,1.0))

    return J
from PIL import Image
def tensor_to_image(tensor):
    tensor=tensor*255
    tensor=np.array(tensor,dtype=np.uint8)

    if np.ndim(tensor)>3:
        tensor=tensor[0]

    return Image.fromarray(tensor)
epochs = 2501
for i in range(epochs):
    train_step(generated_image)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        plt.imshow(image) 
