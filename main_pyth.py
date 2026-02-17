import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19,vgg19
import os
import cv2
import matplotlib.pyplot as plt


CONTENT_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/4. Face Recognition - Neural Style Transfer/Custom Neural Style/images/content/"

STYLE_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/4. Face Recognition - Neural Style Transfer/Custom Neural Style/images/style/"


base_model=VGG19(include_top=False,weights="imagenet",input_shape=(400,400,3))
base_model.trainable=False
print(len(base_model.layers))

middle_layer=base_model.layers[12]
print(middle_layer)
base_model.layers
intermediate_model=Model(inputs=base_model.input,outputs=middle_layer.output)

def compute_content_cost(a_C,a_G):
    
    m,n_h,n_w,n_c=a_G.shape

    a_C_unrolled=tf.reshape(tf.transpose(a_C,[3,1,2,0]),(n_c,-1))
    a_G_unrolled=tf.reshape(tf.transpose(a_G,[3,1,2,0]),(n_c,-1))

    J_content=tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)/(4*n_c*n_h*n_w)

    return J_content


STYLE_LAYERS={
    "block1_conv2":0.5,
    "block2_conv2":1.0,
    "block3_conv2":1.5,
    "block4_conv2":3.0,
    "block5_conv2":4.0
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


def total_cost(J_content,J_style,alpha=10,beta=40):
    
    J_total=alpha*J_content + beta*J_style
    return J_total


content_image=cv2.imread(CONTENT_PATH+"louvre.jpg")
content_image=cv2.cvtColor(content_image,cv2.COLOR_BGR2RGB)
content_image=cv2.resize(content_image,(400,400))
content_image=content_image.astype("float32")/255.
print(content_image.shape)

plt.figure(figsize=(3,3))
plt.imshow(content_image)
plt.axis("off")


style_image=cv2.imread(STYLE_PATH+"monet.jpg")
style_image=cv2.cvtColor(style_image,cv2.COLOR_BGR2RGB)
style_image=cv2.resize(style_image,(400,400))
style_image=style_image.astype("float32")/255.
print(style_image.shape)

plt.figure(figsize=(3,3))
plt.imshow(style_image)
plt.axis("off")

generated_image=np.random.uniform(size=content_image.shape,low=0.0,high=1.0)

print(generated_image.shape)

plt.figure(figsize=(3,3))
plt.axis("off")
plt.imshow(generated_image)
plt.show()

a_C=intermediate_model(np.expand_dims(content_image,axis=0))
print("a_C shape : ",a_C.shape)

def get_layer_outputs(model):

    outputs=[model.get_layer(name).output for name in list(STYLE_LAYERS.keys())]
    
    temp_model=Model(inputs=[model.input],outputs=outputs)
    return temp_model

vgg_layer_outputs=get_layer_outputs(base_model)
a_S_array=vgg_layer_outputs(np.expand_dims(style_image,axis=0))
print(type(a_S_array),len(a_S_array))

optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)

def train_step(generated_image):
    with tf.GradientTape() as tape:

        a_G=intermediate_model(generated_image)
        J_content=compute_content_cost(a_C,a_G)

        a_G_array=vgg_layer_outputs(generated_image)
        J_style=compute_style_cost(a_S_array,a_G_array)

        J=total_cost(J_content,J_style,alpha=10,beta=400)

    grad=tape.gradient(J,generated_image)
    optimizer.apply_gradients([(grad,generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image,0.0,1.0))

    return J

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


generated_image=generated_image.astype(np.float32)
print(generated_image.dtype)

generated_image=tf.Variable(np.expand_dims(generated_image,axis=0))
# print(generated_image)

epochs = 5001
for i in range(epochs):
    loss=train_step(generated_image)
    if i%250==0:
        print(f"epoch {i}")
        print(loss)
        image=tensor_to_image(generated_image)
        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

output_image=generated_image.numpy()
print(output_image.shape)
output_image=output_image[0]
# output_image=cv2.resize(output_image,(800,600))
output_image=(output_image*255).astype("int")

# print(output_image*255)
plt.imshow(output_image)
print(output_image.shape)