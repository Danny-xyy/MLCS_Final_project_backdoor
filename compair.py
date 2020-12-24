

import keras
import keras.backend as K
from keras.utils import to_categorical
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def data_preprocess(x_data):
    return x_data/255

def eval(data_filename,bd_model):
    x_test, y_test = data_loader(data_filename)
    x_test = data_preprocess(x_test)

    

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    return class_accu



#--------------setting addr--------------
print("start test")

bad_net_addr = str(sys.argv[1])               #"models/multi_trigger_multi_target_bd_net.h5"
clean_net_addr= str(sys.argv[2])              #"output_model/multi_trigger_multi_target_clean_net.h5"
input_data_addr=str(sys.argv[3])              # "data/clean_test_data.h5"
save_addr=     str(sys.argv[4])               #"output_label/test.h5"


#---------get data----------------#
def get_data(data_addr_in):

    data = h5py.File(data_addr_in, 'r')

    x_data_out = np.array(data['data'])
    if ('label' in data):
        y_data_out = np.array(data['label'])
    else:
        y_data_out=[]
    x_data_out = x_data_out.transpose((0,2,3,1))
    x_data_out = x_data_out/255

    return x_data_out,y_data_out




#---------------load model-------------------------

bd_model = keras.models.load_model(bad_net_addr)
clean_model = keras.models.load_model(clean_net_addr)
bd_model.summary()
clean_model.summary()

img_index=0

#-------------change---------------
#input_data_addr=bad_data_filename#-----------------------------------------------------

data_x,data_y=get_data(input_data_addr)



data_y_out=[]
print("# of image = ",len(data_x))

for img_index in range( len(data_x) ):

    label_bad_net=np.argmax(bd_model.predict(data_x[img_index:img_index+1]))
    label_clean_net=np.argmax(clean_model.predict(data_x[img_index:img_index+1]))

    if label_bad_net==label_clean_net:
        output=label_clean_net
    else:
        output=1284

    data_y_out.append(output)

print(len(data_y_out))

hf = h5py.File(save_addr, 'w')
hf.create_dataset('data', data=data_x)
hf.create_dataset('label', data=data_y_out)
hf.close()
#---------------out put accuracy---------------------
if data_y!=[]:
    match_number=0
    for label_index in range(len(data_y_out)):
        if data_y_out[label_index] == data_y[label_index]:   #---------------------------------
       # if data_y_out[label_index] == 1284:
            match_number=match_number+1

    print("accuracy ",(match_number/len(data_y_out))*100)




#-------------------------


print("end test")
