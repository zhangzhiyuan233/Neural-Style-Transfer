
# Define the neural_transfer model class

import numpy as np
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import time


class neural_transfer():

    def __init__(self,model):
        self.img_model = model
        self.input_shape = tuple(model.input.shape[1:])
        self.img_encoder = None
        self.W = None
        
    # mathod to process image
    def __get_image(self,image_path):
        n_h, n_w, n_c = self.input_shape 
        image = PIL.Image.open(image_path)
        image_array = np.asarray(image.resize((n_h,n_w)))/255.0
        image_array = image_array.reshape((1,n_h,n_w,n_c))
        return image_array

    #save generated image after training.
    def __save_image(self, path, image):
        image = np.clip(image[0]*255, 0, 255).astype('uint8')
        image = PIL.Image.fromarray(image, 'RGB')
        image.save(path)
        return


    #plot the image after processing
    def show_image(self, image_path):
        image = self.__get_image(image_path)
        plt.imshow(image[0])
        return


    #get the encoder model from pretrained model
    def get_img_encoder(self,content_layer , style_layers):
        self.img_model.trainable = False
        n = len(style_layers)
        self.style_layer_weights = [1/n]*n
        output_layer_C = self.img_model.get_layer(content_layer).output  # use layer block3_conv3 for content representation

        output_layers_S = []

        for layer in style_layers:        #use layers in STYLE_LAYERS for style representations
            output_layers_S.append(self.img_model.get_layer(layer).output)

        new_input = self.img_model.input
        new_output = [output_layer_C] + output_layers_S
        ## let img_encoder denote the pretrained model
        self.img_encoder = tf.keras.Model(new_input, new_output)
        return
    
    
    # content cost
    def __compute_content_cost(self, a_C, a_G):


        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        a_C_unrolled = tf.reshape(a_C, [m,-1,n_C])
        a_G_unrolled = tf.reshape(a_G, (m,-1,n_C))
           
        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)
       
        return J_content
    
    
    #style cost  
    def __gram_matrix(self, A):

        GA = tf.matmul(A, tf.transpose(A)) 
        return GA

    def __compute_layer_style_cost(self,a_S, a_G):
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        

        a_S = tf.reshape(tf.transpose(a_S, perm=[3,1,2,0]), [n_C, -1])
        a_G = tf.reshape(tf.transpose(a_G, perm=[3,1,2,0]), [n_C, -1])

 
        GS = self.__gram_matrix(a_S)
        GG = self.__gram_matrix(a_G)

  
        J_style_layer =  tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*n_C**2*(n_W*n_H)**2)
        
        
        return J_style_layer

    def __compute_style_cost(self,output_S,output_G, weights):
        
        style_cost = 0
        
        for i in range(1, len(output_G)):
            a_S = output_S[i]
            a_G = output_G[i]
            style_cost += weights[i-1]*self.__compute_layer_style_cost(a_S, a_G)
            
        return style_cost
    
    #total cost
    def __total_cost(self, J_content, J_style, alpha, beta):
        J = alpha*J_content+ beta*J_style

        return J
  

    def __cost(self, C, S, style_layer_weights, alpha, beta, noise_ratio):   # C:content image, S:style, model: pretrained model
    
        G = self.__generate_noise_image(C, noise_ratio)   # generate graph
        
        output_C = self.img_encoder(C)
        output_G = self.img_encoder(G)
        output_S = self.img_encoder(S)

        # compute content cost
        a_C, a_G = output_C[0], output_G[0]

        J_content = self.__compute_content_cost(a_C, a_G)
        
        
        # compute style cost
        J_style = self.__compute_style_cost(output_S,output_G, style_layer_weights)

        #total cost
        J = self.__total_cost(J_content, J_style, alpha=alpha, beta=beta)

        return J_content, J_style, J
    

    def __generate_noise_image(self, content_image, noise_ratio):
    
        
        # Generate a random noise_image
        
        # Set the input_image to be a weighted average of the content_image and a noise_image
        input_image = self.W * noise_ratio + content_image * (1 - noise_ratio)
        
        return input_image


    #initialize training weights
    def __initialize_weights(self, seed):
        Initializer = tf.random_uniform_initializer(-1,1,seed=seed)
        n_w, n_h, n_c = self.input_shape
        self.W = tf.Variable(Initializer((1, n_w, n_h, n_c)))
        return
    
    def set_opt(self, opt):
        self.opt = opt
        return
        
    # method to reset weights
    def reinialize_weights(self, seed=123):
        Initializer = tf.random_uniform_initializer(-1,1, seed=seed)
        n_w, n_h, n_c = self.input_shape
        self.W = tf.Variable(Initializer((1, n_w, n_h, n_c)))
        return


    #one step training
    def __one_step(self, C, S, style_layer_weights, alpha, beta, noise_ratio):
        with tf.GradientTape() as tape:
            J_content, J_style, J = self.__cost(C, S, style_layer_weights=style_layer_weights, 
                                                alpha=alpha, beta=beta, noise_ratio=noise_ratio)

        grads = tape.gradient(J, [self.W])
        opt = self.opt
        opt.apply_gradients(zip(grads, [self.W]))

        return J_content, J_style, J

    #main method for training
    def model_nn(self, content_path, style_path, sav_dir ,start = 0 ,num_iterations = 20000, noise_ratio=0.2, alpha = 10, beta = 120, style_layer_weights = None, seed = 123):
        start_time = time.time()
        if not self.img_encoder:
            print("Please run method get_img_encoder to get image encoder first!")
            return 
        if self.W is None:
            self.__initialize_weights(seed=seed)

        if style_layer_weights is None:
            style_layer_weights = self.style_layer_weights

        C = self.__get_image(content_path)
        S = self.__get_image(style_path)

        print(style_layer_weights)
        for i in range(num_iterations):
        

            Jc, Js, Jt = self.__one_step(C = C, S = S, style_layer_weights = style_layer_weights,
                                         alpha = alpha, beta = beta, noise_ratio = noise_ratio)

            generated_image = self.__generate_noise_image(C,noise_ratio = noise_ratio)
  
            if (start+i)<200:
                if i%10 == 0:
                    print("Iteration " + str(start + i) + " :")
                    print("total cost = " + str(Jt.numpy()))
                    print("content cost = " + str(Jc.numpy()))
                    print("style cost = " + str(Js.numpy()))
                    
                    # save current generated image in the "/output" directory
                    self.__save_image(sav_dir+'/' +str(start + i) + ".jpg", generated_image)



            elif i%200 == 0:
                print("Iteration " + str(start + i) + " :")
                print("total cost = " + str(Jt.numpy()))
                print("content cost = " + str(Jc.numpy()))
                print("style cost = " + str(Js.numpy()))
                
                # save current generated image in the "/output" directory
                self.__save_image(sav_dir+'/'+str(start + i) + ".jpg", generated_image)
        
        # save last generated image
        self.__save_image(sav_dir +'/'+str(start + i) +'.jpg', generated_image)
        difference_time = time.time() - start_time
        print('################################')
        print('total runtime for genrating graph:'+str(difference_time))
        print('alpha = '+ str(alpha))
        print('beta = '+ str(beta))
        return 



        
