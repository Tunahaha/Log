import os
import pathlib
from tqdm import tqdm
import threading
import subprocess






def case(num,choose,label):
    model_path = 'dnn_example/cnn_simple_2.h5'
    image_dir = f'dnn_example/shap個數{num}/{choose}' #random跟shap、corner作切換
    
    print(image_dir)
    
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
        print("#"*60, f"第{label}個label", "#"*60)
        filename = f'{label}_{j}_{choose}' #random跟shap作切換
        if filename not in all_input_filename:
            print(f"{filename} doesn't exists")
            print(f"skip {filename}")
            continue
        
        in_filepath = os.path.join(image_dir, f'{filename}.in')
        log_filepath = f'./log/shap個數{num}/{choose}/stack/{filename}.log' #random跟shap作切換(資料夾位址)

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



num=['1','10','30','50']
choose=['random','shap','corner']
label=['0','1','2','3','4','5','6','7','8','9']
threads=[]
for i in num:
    for j in choose:
        for k in label:
            threads.append(threading.Thread(target=case, args=(i,j,k)))


#t=threading.Thread(target=case, args=(50,'corner',3))
#t.start()
#t.join()
# 開始
for t in threads:
    t.start()

# 等待所有子執行緒結束
for t in threads:
    t.join()


# log_filepath = './log/6_112.log'
# test_cmd = 'python3 ./dnnct_wrapper.py dnn_example/cnn_simple_2.h5 dnn_example/cifar10_example/6_112.in'
# test_cmd = test_cmd.split(' ')
# print(test_cmd)

# with open(log_filepath, 'w') as f:
#     subprocess.run(test_cmd, stdout=f)

# print('finish')

