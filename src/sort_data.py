# _*_ coding: utf-8 _*_
# @Time: 2022/1/26 15:57 
# @Author: 韩春龙
import torch
import glob
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
import copy

"""
生成测试数据集2  11489  2001 2001 2001 2001 2000 1485
"""

pts = sorted(glob.glob('../data/cnndm/cnndm.test.' + '*.pt'))
labs = torch.load('../sort_lab.pt')
tokenizer = BertTokenizer.from_pretrained('../temp/bert-base-vocab.txt')
num123 = 0
for pt in pts:
    a = torch.load(pt)
    data_new = []
    for data in tqdm(a):
        src = []
        tgt = []
        src_txt = copy.deepcopy(data["src_txt"])
        tgt_txt = copy.deepcopy(data["tgt_txt"])

        tgt.append(int(1))
        content = tokenizer.tokenize(tgt_txt)
        tgt.extend(tokenizer.convert_tokens_to_ids(content))
        tgt.append(int(2))


        lab = labs[num123]
        num123 += 1

        biaoji = 0
        hiu = False
        for nums, txt_id in enumerate(lab):
            content = tokenizer.tokenize(data["src_txt"][txt_id])
            src.append(tokenizer.cls_token_id)
            src.extend(tokenizer.convert_tokens_to_ids(content))
            src.append(tokenizer.sep_token_id)
            if len(src) > 512:
                biaoji = nums
                hiu = True
                break
        if hiu:
            lab = lab[:biaoji]
            lab.sort()

            src_new = []
            segs_new = []
            for nums, txt_id in enumerate(lab):
                content = tokenizer.tokenize(data["src_txt"][txt_id])

                src_new.append(tokenizer.cls_token_id)
                src_new.extend(tokenizer.convert_tokens_to_ids(content))
                src_new.append(tokenizer.sep_token_id)

                if (nums + 1) % 2 == 0:
                    segs_new.extend([1 for x in range(len(content) + 2)])
                else:
                    segs_new.extend([0 for x in range(len(content) + 2)])

            assert len(src_new) == len(segs_new) and len(src_new) <= 512

            data["src_new"] = src_new
            data["segs_new"] = segs_new
            data["tgt"] = tgt
            data_new.append(data)

        else:
            src_new = []
            segs_new = []
            for nums, txt_id in enumerate(lab):
                content = tokenizer.tokenize(data["src_txt"][txt_id])

                src_new.append(tokenizer.cls_token_id)
                src_new.extend(tokenizer.convert_tokens_to_ids(content))
                src_new.append(tokenizer.sep_token_id)

                if (nums + 1) % 2 == 0:
                    segs_new.extend([1 for x in range(len(content) + 2)])
                else:
                    segs_new.extend([0 for x in range(len(content) + 2)])

            assert len(src_new) == len(segs_new) and len(src_new) <= 512

            data["src_new"] = src_new
            data["segs_new"] = segs_new
            data["tgt"] = tgt
            data_new.append(data)

    torch.save(data_new, pt)
    print("保存数据文件" + pt)

