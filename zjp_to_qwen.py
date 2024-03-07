# -*- coding: utf-8 -*-

import os 
import codecs
import json
import random

main_dir = '/mnt/chongqinggeminiceph1fs/geminicephfs/security-others-common'
local_dir = f'{main_dir}/doodleliang/data_hub/zjp'

cur_id = 0
final_result = []
for file in os.listdir(local_dir):
    if 'aug' in file:
        continue
    cur_file = os.path.join(local_dir, file)
    for line in codecs.open(cur_file, 'r', encoding='utf-8'):
        cur_id += 1
        cur_obj = json.loads(line.strip())
        question = cur_obj['question']
        answer = cur_obj['answer']
        ret_obj = {
            "id": str(cur_id),
            "conversations": [
                {
                    "from": "user",
                    "value": question
                },
                {
                    "from": "assistant",
                    "value": answer
                }
            ]
        }
        final_result.append(ret_obj)
random.shuffle(final_result)
print(f"total {len(final_result)} gen")
with codecs.open('data/zjp_train.json', 'w', encoding='utf-8') as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)
