import re
import argparse
from tqdm import tqdm
import os
import torch
import json
from sklearn.model_selection import train_test_split
from dataset import BasicSample
from utils import create_entity_to_id

def from_offset_to_index(word_list,sentence,start_offset,end_offset,target=None):
    sentence=sentence.lower()
    target=target.lower()
    ##Xử lý trường hợp start offset và end offset là "null":
    if start_offset=="null" and end_offset=="null":
        start_offset=0
        end_offset=len(sentence)
    elif start_offset=="null":
        start_offset=0
    elif end_offset=="null":
        end_offset=len(sentence)


    ##Chane normal string to regrex form
    target_temp=target
    target_temp=re.sub('\[','\[',target_temp)
    target_temp=re.sub('\]','\]',target_temp)
    target_temp=re.sub('\(','\(',target_temp)
    target_temp=re.sub('\)','\)',target_temp)
    target_temp=re.sub('\+','\+',target_temp)
    target_temp=re.sub('\*','\*',target_temp)

    ##Getting real offset indexes because some examples have wrong offset
    real_indexes=[(match.start(),match.end()) for match in re.finditer(target_temp,sentence)]  
    new_indexes=0
    min=abs((0.5*(real_indexes[new_indexes][-1]+real_indexes[new_indexes][0]))-(0.5*(end_offset+start_offset)))
    for i in range(1,len(real_indexes)):
        if abs((0.5*(real_indexes[i][-1]+real_indexes[i][0]))-(0.5*(end_offset+start_offset)))<min:
            new_indexes=i
            min=abs((0.5*(real_indexes[new_indexes][1]+real_indexes[new_indexes][0]))-(0.5*(end_offset+start_offset)))

    ##Update exact indexes
    start_offset=real_indexes[new_indexes][0]
    end_offset=real_indexes[new_indexes][1]


    result=[]
    for idx,token in enumerate(word_list):
        temp_text=' '.join(word_list[:idx+1])
        if len(temp_text)>=start_offset:
            result.append(idx)
            for idy in range(idx,len(word_list)):
                temp_text_2=' '.join(word_list[:idy+1])
                if len(temp_text_2)>=end_offset and idy not in result:
                    result.append(idy)
                    assert word_list[result[0]:result[-1]+1]==target.split()
                    return result
                elif len(temp_text_2)>=end_offset and idy in result:
                    assert word_list[result[0]:result[-1]+1]==target.split()
                    return result
    return result


# def from_offset_to_index(word_list,sentence,start_offset,end_offset,target=None):
#     result=[]
#     print('sentence:',sentence)
#     for idx,token in enumerate(word_list):
#         temp_text=' '.join(word_list[:idx+1])
#         if len(temp_text)>=start_offset:
#             result.append(idx)
#             for idy in range(idx,len(word_list)):
#                 temp_text_2=' '.join(word_list[:idy+1])
#                 if len(temp_text_2)>=end_offset and idy not in result:
#                     result.append(idy)
#                     print(word_list[result[0]:result[-1]+1])
#                     print(target.split())
#                     assert word_list[result[0]:result[-1]+1]==target.split()
#                     return result
#                 elif len(temp_text_2)>=end_offset and idy in result:
#                     print(word_list[result[0]:result[-1]+1])
#                     print(target.split())
#                     assert word_list[result[0]:result[-1]+1]==target.split()
#                     return result
#     return result



def get_triplets(data_paths):
    text_list=[]
    triplets_list=[]
    entity2id,_=create_entity_to_id(data_paths)
    polarity2id={'POSITIVE':0,'NEGATIVE':1}
    for data_path in data_paths:
        with open(data_path,'r') as json_file:
            data=json.load(json_file)
        document=data['document']
        sentences=document['sentences']
        for sentence in sentences:
            triplets=[]
            text=sentence['content'].lower().split()
            text_list.append(text)
            tags=sentence['tags']

            for tag in tags:
                # if tag['target']=="":
                #     continue

                if tag['target']=='NULL' or tag['target']=="":
                    if 'NULL' not in text_list[-1]:
                        text_list[-1].append('NULL')
                    triplet=([len(text_list[-1])-1],entity2id[tag['entity']],polarity2id[tag['polarity']])
                    triplets.append(triplet)
                    #triplets_list.append(triplets)
                    continue

                opinion_ind=from_offset_to_index(text,sentence['content'],tag['start_offset'],tag['end_offset'],tag['target'])
                triplet=(opinion_ind,entity2id[tag['entity']],polarity2id[tag['polarity']])
                triplets.append(triplet)
            triplets_list.append(triplets)
    return triplets_list,text_list


def fusion_triplet(triplet):
  triplet_aspect=[]
  triplet_opinion=[]
  triplet_sentiment=[]
  for t in triplet:
    if t[0] not in triplet_opinion:
      triplet_opinion.append(t[0])
    triplet_aspect.append(t[1])
    triplet_sentiment.append(t[2])
  return triplet_aspect,triplet_opinion,triplet_sentiment


def make_standard(triple_data,dataset_type):
    
    standard_list=[] ##To save standard data
    
    header_fmt='Make standard {:>4s}'
    for triplet in tqdm(triple_data,desc=f'{header_fmt.format(dataset_type.upper())}'):

        aspect_temp=[]
        opinion_temp=[]
        pol_temp=[]
        opi_pol_temp=[]

        for temp_t in triplet:
            op=[temp_t[0][0],temp_t[0][-1],temp_t[2]]
            o=[temp_t[0][0],temp_t[0][-1]]
            p=[temp_t[2]]
            if o not in opinion_temp:
                opi_pol_temp.append(op)
                opinion_temp.append(o)
                pol_temp.append(p)
            a=temp_t[1]
            if a not in aspect_temp:
                aspect_temp.append(a)

        standard_list.append({
            'asp_target':aspect_temp,
            'opi_target':opinion_temp,
            'pol_target':pol_temp,
            'opi_pol_target':opi_pol_temp
        })

    return standard_list

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Processing data')
    ##Define path where save unprocessed data and where to save processed data
    parser.add_argument('--data_paths', type=list, default=['./data/MBBank/data_sentiment_all.json'])
    parser.add_argument('--output_path', type=str, default="./data/MBBank/preprocess")

    args=parser.parse_args()

    ##id to entity dict
    _,id2entity=create_entity_to_id(args.data_paths)

    ##Getting all triplets and texts
    triplets,texts=get_triplets(args.data_paths)

    ##Spliting train, dev, test
    texts_train,texts_temp,triplets_train,triplets_temp=train_test_split(texts,triplets,test_size=0.2,random_state=42)
    texts_dev,texts_test,triplets_dev,triplets_test=train_test_split(texts_temp,triplets_temp,test_size=0.5,random_state=42)

    ##Begin processing flow
    DATASET_TYPE_LIST=['train','dev','test']

    for dataset_type in DATASET_TYPE_LIST:

        if dataset_type=='train':
            text_list=texts_train
            triplet_list=triplets_train
        elif dataset_type=='dev':
            text_list=texts_dev
            triplet_list=triplets_dev
        else:
            text_list=texts_test
            triplet_list=triplets_test

        sample_list=[]
        header_fmt='Processing {:>5s}'
        for i in tqdm(range(len(text_list)),desc=f'{header_fmt.format(dataset_type.upper())}'):
            triplet=triplet_list[i] ##Get one triplet in triple label data
            text=text_list[i]
            already_ta=set()

            ##Creating list of start end pos aspect opinion
            triplet_aspect,triplet_opinion,triplet_sentiment=fusion_triplet(triplet)
            #triplet_opinion=sorted(triplet_opinion,key=lambda x:x[0])

            ##Define some list to save data for training
            aspect_query_list=[]
            opinion_answer_list=[]
            opinion_query_list=[]
            aspect_answer_list=[]
            
            ##Initialize label
            opinion_label=[0]*len(text)
            aspect_label=[0]*len(id2entity)
            sentiment_label=[-1]*len(text)

            ##Initialize query
            aspect_as_query=''
            opinion_as_query=''

            for idx in range(len(triplet_opinion)):
                to=triplet_opinion[idx]
                ta=triplet_aspect[idx]
                s=triplet_sentiment[idx]
                opinion_label[to[0]]=1
                opinion_label[to[0]+1:to[-1]+1]=[2]*len(text[to[0]+1:to[-1]+1])
                opinion_as_query=opinion_as_query+' '.join(text[to[0]:to[-1]+1])+' '
                aspect_label[ta]=1
                if ta not in already_ta:
                    already_ta.add(ta)
                    aspect_as_query=aspect_as_query+id2entity[ta]+' '
                sentiment_label[to[0]:to[-1]+1]=[s]*len(text[to[0]:to[-1]+1])
            opinion_as_query=opinion_as_query[:-1].lower().split()
            aspect_as_query=aspect_as_query[:-1].lower().split()
            aspect_query_list.append(aspect_as_query) 
            opinion_query_list.append(opinion_as_query)
            opinion_answer_list.append(opinion_label)
            aspect_answer_list.append(aspect_label)

            ##Creating ProcessedSample and save to sample_list
            sample = BasicSample(
                text,
                aspect_query_list[0],
                opinion_answer_list[0],
                opinion_query_list[0],
                aspect_answer_list[0],
                sentiment_label
            )
            sample_list.append(sample)

        ##Making standard data
        if dataset_type=='dev':
            dev_standard=make_standard(triplet_list,dataset_type)
        elif dataset_type=='test':
            test_standard=make_standard(triplet_list,dataset_type)

        ##Save the processed data
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        output_path=f'{args.output_path}/{dataset_type}_PREPROCESSED_NEW.pt'
        print(f"Saved data to `{output_path}`.")
        torch.save(sample_list,output_path)
    
    ##Saving standard_data
    output_standard_path=f'{args.output_path}/data_standard_new.pt'

    print(f"Saved data : `{output_standard_path}`.")
    torch.save({
        'dev':dev_standard,
        'test':test_standard
    },output_standard_path)