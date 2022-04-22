import logging
from shutil import ignore_patterns
import torch
import torch.nn.functional as F
import json

def get_logger(filename,verbosity=1,name=None):
  level_dict={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
  formatter=logging.Formatter(
      "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s"
  )
  logger=logging.getLogger(name)
  logger.setLevel(level_dict[verbosity])

  fh=logging.FileHandler(filename,'w')
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  sh=logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger

def create_entity_to_id(file_url_list):
    entity=set()
    result_dict={}
    reverse_result_dict={}

    for file_url in file_url_list:

        with open(file_url,'r') as json_file:
            data=json.load(json_file)

        document=data['document']
        sentences=document['sentences']

        for sentence in sentences:
            text=sentence['content'].split()
            tags=sentence['tags']

            for tag in tags:
                entity.add(tag['entity'])

    for idx,ent in enumerate(sorted(list(entity))):
        result_dict[ent]=idx
        reverse_result_dict[idx]=ent.lower().replace('_',' ').replace('&',' & ')
    
    return result_dict,reverse_result_dict


##This cell contain function to resize tensor for Cross Entropy loss
def normalize_size(tensor):
  ##Hàm chuẩn hóa size tensor lấy lại theo code B-MRC
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_A_loss(logits,targets,ifgpu=True):

    pos_weight=torch.tensor([1.]*22)
    
    if ifgpu==True:
        logits=logits.cuda()
        targets=targets.cuda()
        pos_weight=pos_weight.cuda()
    
    loss=F.binary_cross_entropy_with_logits(logits.float(),targets.float(),pos_weight=pos_weight)

    return loss

def calculate_O_loss(logits,targets,ifgpu=True,ignore_indexes=[],model_type=None):
    ##Hàm này tính loss cho aspect hay opinion
    ##Theo thống kê nhãn 0 nhiều gấp 8 lần nhãn 1 và gấp 16 lần nhãn 2 nên ta sẽ đánh weight theo thứ tự
    ##[1,2,4]
    gold_targets=normalize_size(targets)
    pred=normalize_size(logits)

    if ifgpu==True:
        weight = torch.tensor([1, 2, 4]).float().cuda()
    else:
        weight = torch.tensor([1, 2, 4]).float()

    loss=F.cross_entropy(pred,gold_targets.long(),ignore_index=-1,weight=weight)

    return loss

def is_one_exist(labels,ignore_index):
    '''
        Hàm giúp kiểm tra nếu có nhãn 1 trong labels hay không giúp quyết định bước xây dựng query tiếp theo
    '''
    if 1 not in labels:
        return False
    else:
        count=0
        one_index=(labels==1).nonzero(as_tuple=True)[0]
        for idx in one_index:
            if idx.item() in ignore_index:
                count+=1
        if count==len(one_index):
            return False
    return True