import os
import pathlib
from tqdm import tqdm
import subprocess

model_path = 'dnn_example/cnn_simple_2.h5'
#image_dir = 'dnn_example/shap個數10/shap' #shap
image_dir = 'dnn_example/shap個數10/corner' #corner
#image_dir = 'dnn_example/shap個數10/random' #random
all_input_path = pathlib.Path(image_dir).glob('*.in')
all_input_filename = []
max_num = []

for x in all_input_path:
    all_input_filename.append(x.stem)
    x = int(x.stem.split('_')[1])
    max_num.append(x)

max_num = max(max_num)
print('max_num:', max_num)
print('-'*40, 'start', '-'*40)

for j in range(1, max_num+1):
    for i in range(10):
        print("#"*60, f"第{i}個label", "#"*60)
        #filename = f'{i}_{j}' #shap
        #filename = f'{i}_{j}_random' #random
        filename = f'{i}_{j}_corner' #corner
        if filename not in all_input_filename:
            print(f"{filename} doesn't exists")
            print(f"skip {filename}")
            continue
        
        in_filepath = os.path.join(image_dir, f'{filename}.in')
        #log_filepath = f'./log/shap/stack/{filename}.log' #shap stack
        #log_filepath = f'./log/shap/queue/{filename}.log' #shap queue
        log_filepath = f'./log/corner/stack/{filename}.log' #corner stack
        #log_filepath = f'./log/corner/queue/{filename}.log' #corner queue
        #log_filepath = f'./log/random/stack/{filename}.log' #random stack
        #log_filepath = f'./log/random/queue/{filename}.log' #random queue
        if os.path.exists(log_filepath):
            print(f"log file {log_filepath} exists")
            print(f"skip {filename}")
            continue

        cmd = f'python3 ./dnnct_wrapper.py {model_path} {in_filepath}'
        print('Running:', cmd)

        cmd = cmd.split(' ')
        with open(log_filepath, 'w') as f:
            subprocess.run(cmd, stdout=f)
            print('#'*80)



# log_filepath = './log/6_112.log'
# test_cmd = 'python3 ./dnnct_wrapper.py dnn_example/cnn_simple_2.h5 dnn_example/cifar10_example/6_112.in'
# test_cmd = test_cmd.split(' ')
# print(test_cmd)

# with open(log_filepath, 'w') as f:
#     subprocess.run(test_cmd, stdout=f)

# print('finish')

