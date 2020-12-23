import keras
import keras.backend as K
from keras.utils import to_categorical
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def eval(data_filename,bd_model):
    x_test, y_test = data_loader(data_filename)
    x_test = data_preprocess(x_test)

    

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    return class_accu


def bubble(input_list):     #  bubble sort
    number = len(input_list[0])
    
    list_out=np.zeros((2,number))
   
    for i in range(number):
        smallest_mean=1000000
        neuron_index=0
        list_index=0
        for j in range(number-i):

            if input_list[1,j+i]<smallest_mean:
                smallest_mean = input_list[1,j+i]
                neuron_index = input_list[0,j+i]
                list_index=j+i

        input_list[0,list_index]=input_list[0,i]
        input_list[1,list_index]=input_list[1,i]

        list_out[0,i]=neuron_index
        list_out[1,i]=smallest_mean
    return list_out

#--------------setting addr--------------
print("start test")
clean_data_filename="data/clean_validation_data.h5"
clear_test_data="data/clean_test_data.h5"
bad_net_addr = "models/anonymous_2_bd_net.h5"
saving_addr = "output_model/anonymous_2_clean_net.h5"
layer_name='conv_3'
#model_filename="models\sunglasses_bd_net.h5"

#---------get data----------------#
def get_data(data_addr_in):

    data = h5py.File(data_addr_in, 'r')

    x_data_out = np.array(data['data'])
    y_data_out = np.array(data['label'])
    x_data_out = x_data_out.transpose((0,2,3,1))
    x_data_out = x_data_out/255

    return x_data_out,y_data_out

clean_data_x,clean_data_y=get_data(clean_data_filename)

#---------------load model-------------------------

bd_model = keras.models.load_model(bad_net_addr)
bd_model.summary()

print("#-----------------------------------")
layer_numbers=len(bd_model.layers)
print("layer number : ",layer_numbers)
print("#-----------------------------------")
init_acc = eval(clear_test_data,bd_model)
#---------------creat new model-----------------------------


my_output_layer=bd_model.get_layer(layer_name).output

new_model = keras.Model(inputs=bd_model.input, outputs=my_output_layer)

layer_output=new_model.predict(clean_data_x[0:1])

neurons_number=len(layer_output[0,0,0])




#----------------------input image and get a list for pruning-------------
def get_layer_output(x_data_in,model_in,neurons_number_in):

    image_number=len(x_data_in)
    neuron_mean_all=np.zeros((2,neurons_number_in))


    for image_index in range(image_number):
        print("\r image number :",image_index,end='')
        layer_output=model_in.predict(x_data_in[image_index:image_index+1])

        neuron_mean_sub=np.zeros((2,neurons_number_in))
    
        for i in range(neurons_number_in):
            mean_value=np.mean(layer_output[0,:,:,i])
            
            neuron_mean_sub[0,i]=i
            neuron_mean_sub[1,i]=mean_value

        if (image_index==0):
            neuron_mean_all=neuron_mean_sub
        else:
            neuron_mean_all[1,:]=neuron_mean_all[1,:]+neuron_mean_sub[1,:]

    print(" ")
    neuron_mean_all[1,:]=neuron_mean_all[1,:]/image_number

    return neuron_mean_all

output_list_clean = get_layer_output(clean_data_x,new_model,neurons_number)

output_list=bubble(output_list_clean)

#-----prun--------------------------------------------------------------------
weights_orig,bias_orig=bd_model.get_layer(layer_name).get_weights()
neuron_idx=0
print("neurons in layer = ", neurons_number)
neuron_values = []
neuron_values.append(neurons_number-2)
neuron_values.append(neurons_number/2)
step_size = int(neurons_number/2)
perf_threshold = init_acc
neuron_list = []
acc_list = []
neuron_list.append(0)
acc_list.append(init_acc)

neurons_number_to_change=neurons_number-max(int(neurons_number/20),3)
new_weights=np.copy(weights_orig)
new_bias=np.copy(bias_orig)



new_weights[:,:,:,:neurons_number_to_change]=0*new_weights[:,:,:,:neurons_number_to_change]
new_bias[:neurons_number_to_change]=0*new_bias[:neurons_number_to_change]

bd_model.get_layer(layer_name).set_weights( (new_weights,new_bias) )
bd_model.fit(clean_data_x, clean_data_y, epochs=2)
print("neurons_number_to_change = ",neurons_number_to_change)
low_acc = eval(clear_test_data,bd_model) 
print("Perf threshold = ", perf_threshold)
perf_threshold = perf_threshold - (perf_threshold - low_acc)/2
neuron_list.append(neurons_number_to_change)
acc_list.append(low_acc)
print(low_acc)

print("Perf threshold = ", perf_threshold)

if(init_acc - 5 <= perf_threshold):
    perf_threshold = init_acc - 5
print("Perf threshold = ", perf_threshold)

neuron_idx = int(neurons_number/2)

#Coarsed-Grained Pruning
#while( (neurons_number_to_change<neurons_number) and ( eval(clear_test_data,bd_model)>perf_threshold ) ):
while(step_size>2):

    #neurons_number_to_change=neurons_number_to_change+1
    neurons_number_to_change=neuron_idx


    new_weights=np.copy(weights_orig)
    new_bias=np.copy(bias_orig)



    new_weights[:,:,:,:neurons_number_to_change]=0*new_weights[:,:,:,:neurons_number_to_change]
    new_bias[:neurons_number_to_change]=0*new_bias[:neurons_number_to_change]

    bd_model.get_layer(layer_name).set_weights( (new_weights,new_bias) )
    bd_model.fit(clean_data_x, clean_data_y, epochs=2)
    print("neurons_number_to_change = ",neurons_number_to_change)
    curr_acc = eval(clear_test_data,bd_model)
    print(curr_acc)
    
    neuron_list.append(neurons_number_to_change)
    acc_list.append(curr_acc)
    if(curr_acc > perf_threshold):
        neuron_idx = neuron_idx + int(step_size/2)
    else:
        neuron_idx = neuron_idx - int(step_size/2)
    step_size = int(step_size/2)

print("Neuron index = ", neuron_idx)
print(neuron_list)
print(acc_list)
neuron_list_final = []
acc_list_final = []

#Fine-Grained Pruning
for neuron in range(neuron_idx-3,neuron_idx+2):
    #neurons_number_to_change=neurons_number_to_change+1
    neurons_number_to_change=neuron
    if neuron in neuron_list:
       print("Skipping ", neuron)
       neuron_list_final.append(neuron)
       acc_list_final.append(acc_list[neuron_list.index(neuron)])
       continue


    new_weights=np.copy(weights_orig)
    new_bias=np.copy(bias_orig)



    new_weights[:,:,:,:neurons_number_to_change]=0*new_weights[:,:,:,:neurons_number_to_change]
    new_bias[:neurons_number_to_change]=0*new_bias[:neurons_number_to_change]

    bd_model.get_layer(layer_name).set_weights( (new_weights,new_bias) )
    bd_model.fit(clean_data_x, clean_data_y, epochs=2)
    print("neurons_number_to_change = ",neurons_number_to_change)
    curr_acc = eval(clear_test_data,bd_model)
    print(curr_acc)
    neuron_list_final.append(neurons_number_to_change)
    acc_list_final.append(curr_acc)

print(neuron_list_final)
print(acc_list_final)

#Neuron Selection
neuron_pruning = neuron_list_final[acc_list_final.index(max(acc_list_final))]
neurons_number_to_change=neuron_pruning

new_weights=np.copy(weights_orig)
new_bias=np.copy(bias_orig)

new_weights[:,:,:,:neurons_number_to_change]=0*new_weights[:,:,:,:neurons_number_to_change]
new_bias[:neurons_number_to_change]=0*new_bias[:neurons_number_to_change]

bd_model.get_layer(layer_name).set_weights( (new_weights,new_bias) )
bd_model.fit(clean_data_x, clean_data_y, epochs=5)


#---------------------fine tuning
print(neurons_number_to_change)


print(eval(clear_test_data,bd_model))



bd_model.save(saving_addr)
K.clear_session()


#-------------------------


print("end test")