├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
├── output_models                   //this is repaired clean network
    └── sunglasses_clean_net.h5
    └── multi_trigger_multi_target_clean_net.h5
    └── anonymous_1_clean_net.h5
    └── anonymous_2_clean_net.h5
├── output_label                   //this input date set with predicted label
    └── anonymous_1_net_clean_input.h5
    └── anonymous_1_net_poisoned_input.h5
    └── multi_trigger_net_clean_input.h5
    └── multi_trigger_net_eyebrows_poisoned_input.h5
    └── multi_trigger_net_sunglasses_poisoned_input.h5
    └── sunglasses_net_clean_input.h5
    └── sunglasses_net_poisoned_input.h5

└── main.py // this is fine-pruning code to generate repaired clean network
└── compair.py // this is code to generate out put label
└── analysis.py // this is code to analysis the performance of fine-pruning


#I. Dependencies

Python 3.6.12
Keras 2.3.1
Numpy 1.19.2
Matplotlib 3.32
H5py 2.10.0
TensorFlow 2.1.0

#II. Generate repaired clean network

1. In "main.py"  change code line from 56-59, which is the address for clean validation data, clean test data, bad model, and repaired clean network saving address.
2. run the "main.py" code:
	python3 main.py
we already generate repaired clean network and saving in "output_models " folder

#III. Generate output label

1. In "compair.py "  change code line from 31-34,which is the address for backdoor net, clean net, inputdata and saving address of output label

2. In this code:
 	1)if your input data is only image , "compair.py " will saving a '.h5' file with input data ('data') and predict label ('label').
	1)if your input data is image with label, "compair.py " will saving a '.h5' file with input data ('data') and predict label ('label'), and return the predict accuracy.
	Please set the label of poisoned data to 1284 (which is N+1)

3. run the "compair.py" code:
	python3 compair.py
we already generate output data and saving in "output_label " folder
