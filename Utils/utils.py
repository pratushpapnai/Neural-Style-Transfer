import tensorflow as tf


#Calculate Content Cost
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

#Calculate Style Cost
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



#Calculate Total Cost
def total_cost(J_content,J_style,alpha=10,beta=40):
    
    J_total=alpha*J_content + beta*J_style
    return J_total