import os
import pathlib
from tqdm import tqdm
import subprocess

model_path = 'cifar10_example/cnn_simple_2.h5'
image_dir = 'cifar10_example' #random跟shap作切換

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

for j in tqdm(range(5,max_num+1)):
    for i in tqdm(range(7,10)):
        filename = f'{i}_{j}' #random跟shap作切換
        if filename not in all_input_filename:
            continue
        
        in_filepath = os.path.join(image_dir, f'{filename}.in')
        log_filepath = f'./log/shap個數10/shap/{filename}.log' #random跟shap作切換(資料夾位址)
        cmd = f'python3 ./dnnct_wrapper.py {model_path} {in_filepath}'
        print('Running:', cmd)

        cmd = cmd.split(' ')
        with open(log_filepath, 'w') as f:
            subprocess.run(cmd, stdout=f)



# log_filepath = './log/6_112.log'
# test_cmd = 'python3 ./dnnct_wrapper.py dnn_example/cnn_simple_2.h5 dnn_example/cifar10_example/6_112.in'
# test_cmd = test_cmd.split(' ')
# print(test_cmd)

# with open(log_filepath, 'w') as f:
#     subprocess.run(test_cmd, stdout=f)

# print('finish')

