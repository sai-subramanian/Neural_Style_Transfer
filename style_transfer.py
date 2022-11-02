import random
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow import *
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
from tensorflow.keras import optimizers
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#excess = not by me

################## for correcting no algorithm worker error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##################


# from tensorflow.keras import
img_shape = 400

content_img = np.array(Image.open("content.jpg").resize((img_shape, img_shape)))
style_img = np.array(Image.open("style2.jpg").resize((img_shape, img_shape)))
print("code started")
def img_to_tensor(img):
    converted = convert_to_tensor(img , float32)
    return converted
content_img = img_to_tensor(content_img)
style_img = img_to_tensor(style_img)

vgg = VGG19(include_top=False,input_shape=(img_shape, img_shape, 3),weights='imagenet')
# vgg = keras.applications.VGG19(include_top=False, weigths='imagenet', input_shape=(img_shape, img_shape, 3))

vgg.trainable = False
print("model loaded")


def content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = reshape(a_C, shape=[1, -1, n_C])
    a_G_unrolled = reshape(a_G, shape=[1, -1, n_C])

    content_cost = reduce_sum(square(subtract(a_G_unrolled, a_C_unrolled))) / (4 * n_H * n_W * n_C)

    return content_cost


def correlation_across_channels(A):
    style_matrix = matmul(A, transpose(A))  # gram matrix computes style or correlation across the channels of
    # activations of a layer

    return style_matrix


def style_cost_perlayer(style_output, generated_output):
    _, n_H, n_W, n_C = generated_output.get_shape().as_list()
    a_S = style_output
    a_G = generated_output

    #a_S = reshape(a_S,shape=[n_C, -1])  # we are reshaping the activations so that we get a 2D matrix which we can send to -
    #a_G = reshape(a_G,shape=[n_C,-1])  # correlations computing function which will in turn compute the correlation across different
    # channels of a particular layer of a particular image
    a_S  = transpose(reshape(a_S,shape=[-1,n_C]))
    a_G = transpose(reshape(a_G,shape=[-1,n_C]))
    GS = correlation_across_channels(a_S)
    GG = correlation_across_channels(a_G)

    J_style = reduce_sum(square(subtract(GS, GG))) / (4 * (n_H * n_W * n_C) ** 2)

    return J_style

########################################################################################################### for checking excess
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = style_cost_perlayer(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = keras.Model([vgg.input], outputs)
    return model

content_layer = [('block5_conv4', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)


#preprocessed_content =  Variable(image.convert_image_dtype(content_img, float32))
preprocessed_content = expand_dims(content_img, axis=0)
a_C = vgg_model_outputs(preprocessed_content)

#style_img = np.expand_dims(style_img, axis=0)# so that style img has (None,400,400,3) dim insted of (400,400,3)
preprocessed_style = Variable(image.convert_image_dtype(style_img, float32))
preprocessed_style = expand_dims(content_img, axis=0)
a_S = vgg_model_outputs(preprocessed_style)

def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
###########################################################################################################



def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha * J_content + beta * J_style  # here alpha and beta determine how much weightage should be given to content and style in -
    # - the generated image respectively
    return J


# ************************************************** doubtful
def get_activations(img):
    #img = convert_to_tensor(img, dtype=float32)
    output = vgg.layers[-2].output
    model = keras.Model([vgg.inputs], output)
    print("line in get activations cmp")

    print(np.shape(img))
    img = np.expand_dims(img, axis=0)
    print(np.shape(img))

    G_encoding = model(img)
    return G_encoding


# ***************************************************


#print(np.shape(content_img))

#C_encoding = get_activations(content_img)
#S_encoding = get_activations(style_img)
print("ip image activations revieved")

## computing the gradients for optimizations algo like adam
optimizer = optimizers.Adam(learning_rate=0.01)



def train_one_step(generated_image,i):
    print("101 line reached")
    #generated_image = Variable(generated_image,trainable=True) # this line was not allowing the gradients to change
    with GradientTape() as tape:
        tape.watch(generated_image)

        #G_encoding = get_activations(generated_image)
        #J_content = content_cost(C_encoding, G_encoding)

        #J_style = style_cost_perlayer(S_encoding, G_encoding)
        #if(i==0):
            #preprocessed_G = Variable(image.convert_image_dtype(generated_image, float32)) #this line was responsible for the grads to become none as it took computation out of tf env
            #preprocessed_G = Variable(np.expand_dims(preprocessed_G, axis=0))
        #preprocessed_G = img_to_tensor(generated_image)
        preprocessed_G = expand_dims(generated_image,axis=0)
        #else:
            #preprocessed_G = expand_dims(generated_image,axis=0) #Variable(np.expand_dims(generated_image, axis=0))
        a_G = vgg_model_outputs(preprocessed_G)
        J_content = content_cost(a_C,a_G)
        J_style = compute_style_cost(a_S,a_G)
        J = total_cost(J_content, J_style)

    grads = tape.gradient(J, generated_image)#unconnected_gradients='zero')
    print(np.shape(grads))
    print(np.shape(generated_image))

    #optimizer.minimize(J, generated_image,tape = tape)
    #optimizer.apply_gradients(grads_and_vars = ([grads, generated_image]))
    optimizer.apply_gradients(zip([grads], [generated_image]))
    print(J)
    #generated_image.assign(generated_image)
    return J


generated_image = Variable(convert_to_tensor(content_img, dtype=float32))
noise = random.uniform(shape(generated_image), -0.25, 0.25)
generated_image = add(noise, generated_image)
#generated_image = clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
print(generated_image.shape)
imshow(generated_image.numpy())
plt.show()
generated_image = Variable(generated_image,trainable=True)

print("generated image init")


########################     excess
def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


########################excess end
###### training for epochs
print("training started")
epochs = 2501
for i in range(epochs):
    train_one_step(generated_image,i)
    print(f"Epoch {i} ")

    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 50 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        #image.save(f"output/image_{i}.jpg")
        plt.show()
