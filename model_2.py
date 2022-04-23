import torch.nn as nn
from transformers import DebertaV2Tokenizer, DebertaV2Model
from utils import is_one_exist, calculate_A_loss, calculate_O_loss, create_entity_to_id
from dataset_support import generating_next_query
import torch
import torch.nn.functional as F

##Feed Forward Neural Network
class FFNN(nn.Module):
  '''
    Module question answering, bao gồm:
      + Mọt lớp pre-trained model (trong notebook này là bdi-debertav3-xxsmall)
      + Một lớp Linear với hai mode:
        - Một cái dành cho trả lời câu hỏi để tìm aspect
        - Cái còn lại dành cho trả lời câu hỏi để tìm opinion
  '''
  def __init__(self,args):
    hidden_size=args.hidden_size
    entity_size=args.entity_size
    super(FFNN,self).__init__()

    ##Khởi tạo tokenizer theo model pretrained đang sử dụng
    self._embedding=DebertaV2Model.from_pretrained(args.model_type)

    print(f"Loaded `{args.model_type}` model !")
    
    ##FFNN cho việc gán nhãn asp và opi
    self.asp_ffnn=nn.Linear(hidden_size,entity_size)
    self.opi_ffnn=nn.Linear(hidden_size,3)

  def forward(self,input_ids=[], attention_mask=[], token_type_ids=[],answer='aspect'):

    ##Cho input qua bert để lấy hidden state của các token
    hidden_states = self._embedding(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[0]

    ##Logits calculation
    if answer=='aspect':
      logits=self.asp_ffnn(hidden_states[:,0,:])
    elif answer=='opinion':
      logits=self.opi_ffnn(hidden_states)

    return hidden_states,logits

class RoleFlippedModule(nn.Module):
  def __init__(self,args):
    '''
      Module để thay đổi giữa query aspect và query opinion theo args.T vòng, vòng số 0 luôn là initial
    '''
    super(RoleFlippedModule,self).__init__()
    self._model=FFNN(args)  ##Thành phần cho chiều A2O
    self._model2=FFNN(args) ##Thành phần cho chiều O2A

    if args.ifgpu==True and torch.cuda.is_available():
      self._model.cuda()
      self._model2.cuda()
    
    ##Giá trị của sep_id và cls_id với mỗi loại mô hình
    self.sep_id=2
    self.cls_id=0

    self.args=args
  
  def forward(self,batch_dict,model_mode='train'):
    '''
      Các biến cur_answer sẽ lật qua lật lại để role flipped
        + Todo: Tương lai thiết kế có thể ngắt 1 trong hai phần A2O hoặc O2A
    '''
    lossA=0
    lossO=0
    #A2O
    ##Initial
    A2O_aspect_hidden_states,aspect_logits=self._model(batch_dict['initial_input_ids'],batch_dict['initial_attention_mask'],batch_dict['initial_token_type_ids'],answer='aspect')
    
    ##Nếu model đang trong quá trình train mới tính loss
    if model_mode=='train':
      lossA+=self.args.lambda_aspect*calculate_A_loss(aspect_logits,batch_dict['initial_aspect_answers'],ifgpu=self.args.ifgpu)
    
    ##Khởi tạo query cho bước tiếp theo
    input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,batch_dict['initial_input_ids'],self.args,query_type='aspect',model_mode=model_mode)
    
    ##Multihop turn
    cur_answer='opinion'
    for i in range(self.args.T):

      ##Nếu câu trả lời đang tìm hiện tại là opinion
      if cur_answer=='opinion':
        queries_for_opinion=input_ids
        A2O_opinion_hidden_states,opinion_logits=self._model(input_ids,attention_mask,token_type_ids,answer=cur_answer)

        ##Nếu đang trong bước train thì tính loss
        if model_mode=='train': ##Tương tự chỉ tính loss cho model khi ở mode train (*)
          lossO+=self.args.lambda_opinion*calculate_O_loss(opinion_logits,answers,ifgpu=self.args.ifgpu,ignore_indexes=batch_dict['ignore_indexes'],model_type=self.args.model_type)
        
        ##Khởi tạo query cho bước kế tiếp và đổi chiều câu trả lời
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,queries_for_opinion,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='aspect'
      
      ##Câu trả lời đang tìm là aspect
      elif cur_answer=='aspect':
        queries_for_aspect=input_ids
        A2O_aspect_hidden_states,aspect_logits=self._model(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        
        ##Nếu ỏ bước train thì tính loss
        if model_mode=='train':##(*)
          lossA+=self.args.lambda_aspect*calculate_A_loss(aspect_logits,answers,ifgpu=self.args.ifgpu)
        
        ##Khởi tạo query cho bước tiếp và đổi chiều câu trả lời
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,queries_for_aspect,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='opinion'
    
    A2O_aspect_hidden_states,A2O_aspects_list,A2O_opinion_hidden_states,A2O_opinions_list=self.processOutput(A2O_aspect_hidden_states,A2O_opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict,model_mode=model_mode)
    
    aspect_logits=[]
    opinion_logits=[]
    
    
    
    #O2A
    ##Initial
    O2A_opinion_hidden_states,opinion_logits=self._model2(batch_dict['initial_input_ids'],batch_dict['initial_attention_mask'],batch_dict['initial_token_type_ids'],answer='opinion')
    
    ##Nếu đang ở quá trình train thì tính loss
    if model_mode=='train':##(*)
      lossO+=self.args.lambda_opinion*calculate_O_loss(opinion_logits,batch_dict['initial_opinion_answers'],ifgpu=self.args.ifgpu,ignore_indexes=batch_dict['ignore_indexes'],model_type=self.args.model_type)
    
    ##Khởi tạo query cho bước tiếp theo
    input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,batch_dict['initial_input_ids'],self.args,query_type='opinion',model_mode=model_mode)
    ##Multihop turn
    cur_answer='aspect'
    for i in range(self.args.T):

      ##Câu trả lời hiện tại là aspect
      if cur_answer=='aspect':
        queries_for_aspect=input_ids
        O2A_aspect_hidden_states,aspect_logits=self._model2(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        
        ##Nếu là quá trình train thì tiến hành tính loss
        if model_mode=='train': ##(*)
          lossA+=self.args.lambda_aspect*calculate_A_loss(aspect_logits,answers,ifgpu=self.args.ifgpu)
        
        ##Khởi tạo query cho bước kế tiếp và đổi chiều hop
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,aspect_logits,queries_for_aspect,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='opinion'
      
      ##Câu trả lời hiện tại cần tìm là opinion
      elif cur_answer=='opinion':
        queries_for_opinion=input_ids
        O2A_opinion_hidden_states,opinion_logits=self._model2(input_ids,attention_mask,token_type_ids,answer=cur_answer)
        
        ##Nếu trong quá trình train thì tính loss
        if model_mode=='train':##(*)
          lossO+=self.args.lambda_opinion*calculate_O_loss(opinion_logits,answers,ifgpu=self.args.ifgpu,ignore_indexes=batch_dict['ignore_indexes'],model_type=self.args.model_type)
        
        ##Khởi tạo query cho bước tiếp theo và đổi chiều hop
        input_ids,attention_mask,token_type_ids,answers=generating_next_query(batch_dict,opinion_logits,queries_for_opinion,self.args,query_type=cur_answer,model_mode=model_mode)
        cur_answer='aspect'
    
    O2A_aspect_hidden_states,O2A_aspects_list,O2A_opinion_hidden_states,O2A_opinions_list=self.processOutput(O2A_aspect_hidden_states,O2A_opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict,model_mode=model_mode)
    
    result={
        'A2O_aspect_hidden_states':A2O_aspect_hidden_states,
        'A2O_opinion_hidden_states':A2O_opinion_hidden_states,
        'A2O_aspects_list':A2O_aspects_list,
        'A2O_opinions_list':A2O_opinions_list,
        'O2A_aspect_hidden_states':O2A_aspect_hidden_states,
        'O2A_opinion_hidden_states':O2A_opinion_hidden_states,
        'O2A_aspects_list':O2A_aspects_list,
        'O2A_opinions_list':O2A_opinions_list,
        'lossA':lossA,
        'lossO':lossO,
        'ignore_indexes':batch_dict['ignore_indexes']
    }
    
    
    ##If in training process we should add ground truth sentiment labels to result for calculate sentiment loss next step
    if model_mode=='train':
      result['sentiment_labels_list']=batch_dict['sentiments']

    return result

  def processOutput(self,aspect_hidden_states,opinion_hidden_states,aspect_logits,opinion_logits,queries_for_aspect,queries_for_opinion,batch_dict=None,model_mode='train'):
    '''
      Hàm hỗ trợ Module xử lý dữ liệu đầu ra từ logits thành list các aspect và opinion dự đoán được qua
       input.
      Hàm còn hỗ trợ cắt lấy các hidden states chính xác của câu input (như đã biết câu query được padding nên
      hàm này sẽ chỉ lấy đúng hidden states theo vị trí đúng của các token trong câu input trong context của
      thành phần question) 
    '''
    ##Define list to save data output
    aspect_list=[]
    opinion_list=[]
    aspect_hidden_states_list=[]
    opinion_hidden_states_list=[]

    
    for i in range(len(aspect_logits)):
      ##Getting ignore indexes aka sub-word indexes
      index=torch.tensor(batch_dict['ignore_indexes'][i])
      ignore_index=(index == -1).nonzero(as_tuple=True)[0]
      
      ##Opinion
      opinions=[]
      probability=0

      ##Trích lấy vị trí context trong câu query
      passenge_index = (queries_for_opinion[i]==self.sep_id).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long).unsqueeze(1)
      
      ##logit ở vị trí thứ i của batch
      logits=opinion_logits[i]
      
      ##Lấy nhãn opinion theo vị trí của context
      opinion_prob=F.softmax(logits,dim=-1)
      prob_val,prob_label=torch.max(opinion_prob,dim=-1)
      passenge_labels=prob_label[passenge_index].squeeze(1)
      passenge_prob_vals=prob_val[passenge_index].squeeze(1)
      prob_list,index_list=torch.sort(passenge_prob_vals,descending=True)


      ##Xử lý khi nhãn 1 không có trong dự đoán, tiến hành như trên kia
      if is_one_exist(passenge_labels,ignore_index)==False:

        ##Quá trình train thì sử dụng teacher forcing
        if model_mode=='train':
          ##In training process if doesn't find any B label we use grouth truth to learning
          ##Teacher forcing
          passenge_labels=torch.tensor(batch_dict['opinion_answers'][i])
          one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]

        ##Trường hợp không trong quá trình train
        else:
          _opinion_prob=opinion_prob.transpose(0,1)[1]
          passenge_opinion_prob=_opinion_prob[passenge_index].squeeze(1)
          _,one_index=torch.sort(passenge_opinion_prob,descending=True)
          index_list=one_index
        '''if 0 not in passenge_labels:
          two_index=torch.tensor([])
        else:
          two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]'''
        ##Trường hợp này nhãn hai sẽ là rỗng
        two_index=torch.tensor([])

      ##Trường hợp có nhãn 1 trong tập nhãn do mô hình gán
      else:
        one_index=(passenge_labels == 1).nonzero(as_tuple=True)[0]
        two_index=(passenge_labels == 2).nonzero(as_tuple=True)[0]
      count=0

      ##Vòng lặp lấy các opinion
      for j in range(len(index_list)):
        idx=index_list[j].item()

        ##Bỏ qua khi vị trí vừa vào có nhãn hai hoặc là sub-word
        if idx in two_index or idx in ignore_index:
          continue

        ##Tiến hành lấy opinion khi nhãn là 1
        if idx in one_index:
          opinion=[idx]
          probability=prob_list[idx].item()
          count+=1
          idx+=1
          while idx<len(passenge_index) and (idx in two_index or idx in ignore_index):
            opinion.append(idx)
            probability=probability+prob_list[idx].item()
            idx+=1
          
          probability=probability/(len(opinion))
          if probability>=self.args.opinion_threshold:
            opinions.append(opinion)
          else:
            count=count-1


          ##Điều kiện dừng
          if count>self.args.q:
            break
        else:
          continue

      ##Thêm các opinion của hiện tại vào danh sách opinion của cả batch  
      opinion_list.append(opinions)


      ##Aspects
      aspects=[]

      ###Trích  lấy logits
      logits=aspect_logits[i]

      ###Tính toán xác suất
      F_prob=F.sigmoid(logits)
      asp_labels=torch.round(F_prob)
      _,indices=torch.sort(F_prob)

      if 1 not in asp_labels:

        ###Trường hợp đang trong quá trình train thì sử dụng teacher forcing
        if model_mode=='train':
          aspect_labels=torch.tensor(batch_dict['aspect_answers'][i])
          one_labels=(aspect_labels == 1).nonzero(as_tuple=True)[0]

        ##Trường hợp không trong quá trình train
        else:
          one_labels=indices
      
      ##Trường hợp có nhãn 1 trong nhãn:
      else:
        one_labels=(asp_labels==1).nonzero(as_tuple=True)[0]

      ##Getting aspects
      count=0
      for j in indices:
        indice=j.item()
        if indice in one_labels:
          aspects.append(indice)
          count+=1

        ##Điều kiện dừng là số aspect lớn hơn bằng args.p
        if count>=self.args.p:
          break
      
      ##Thêm các aspect của hiện tại vào danh sách aspect của cả batch  
      aspect_list.append(aspects)


      ##Opinion Hidden states
      passenge_index = (queries_for_opinion[i]==self.sep_id).nonzero(as_tuple=True)[0]
      passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long)
      opinion_hidden_states_list.append(opinion_hidden_states[i,passenge_index,:])
      
      ##Aspect Hidden states
      # passenge_index = (queries_for_aspect[i]==self.sep_id).nonzero(as_tuple=True)[0]
      # passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long)
      ##First test using only hidden state of CLS for next sentiment classification step
      aspect_hidden_states_list.append(aspect_hidden_states[i,0,:])
    
    return aspect_hidden_states_list,aspect_list,opinion_hidden_states_list,opinion_list

##This cell contain function for Matching Module
class MatchingModule(nn.Module):
  '''
    Module áp dụng attention mechanism để tìm ra những opinion terms cao điểm nhất với mỗi aspect term dự
     đoán được sau RoleFlipped.
  '''
  def __init__(self,args):
    super(MatchingModule,self).__init__()
    self.sent_ffnn_A2O=nn.Linear(args.hidden_size*2,2) ##Thành phần dán nhãn sentiment cho chiều A2O
    self.sent_ffnn_O2A=nn.Linear(args.hidden_size*2,2) ##Thành phần dán nhãn sentiment cho chiều O2A

    if args.ifgpu==True:
      self.sent_ffnn_A2O.cuda()
      self.sent_ffnn_A2O.cuda()
    
    ##Nếu model là deberta thì cần tokenizer để xử lý dữ liệu đầu ra
    if 'deberta' in args.model_type:
      self._tokenizer=DebertaV2Tokenizer.from_pretrained(args.model_type)
      
    self.args=args

  def forward(self,result_dict,batch_dict,model_mode='train'):

    ##Các danh sách lưu những kết quả đầu ra cuối cùng
    predicts_list=[]
    _aspects_list=[]
    _opinions_list=[]

    ##Khởi tạo loss cho việc gán nhãn cảm xúc
    lossS=0
    for i in range(len(result_dict['A2O_aspects_list'])):

      ##Trích lấy những vị trí của sub-words
      index=torch.tensor(batch_dict['ignore_indexes'][i])
      ignore_index=(index == -1).nonzero(as_tuple=True)[0]
      
      ##Khởi tạo danh sách để lưu opinion (first test only opinion has index)
      ##aspects_list=[]
      opinions_list=[]

      ##A2O
      A2O_aspect_terms=result_dict['A2O_aspects_list'][i]
      A2O_opinion_terms=result_dict['A2O_opinions_list'][i]
      A2O_aspect_hidden_states=result_dict['A2O_aspect_hidden_states'][i]
      A2O_opinion_hidden_states=result_dict['A2O_opinion_hidden_states'][i]
      final_hidden_states=self.matching(A2O_aspect_hidden_states,A2O_opinion_hidden_states,A2O_aspect_terms,A2O_opinion_terms,calc_type='all',direct='A2O',mode='only-cls')
      A2O_logits=[]
      for idx in range(len(final_hidden_states)):
        row=final_hidden_states[idx]
        if torch.sum(row).item()==0 or idx in ignore_index:
          A2O_logits.append([0]*2)
        else:
          A2O_logits.append(self.sent_ffnn_A2O(row).tolist())
      
      ##O2A
      O2A_aspect_terms=result_dict['O2A_aspects_list'][i]
      O2A_opinion_terms=result_dict['O2A_opinions_list'][i]
      O2A_aspect_hidden_states=result_dict['O2A_aspect_hidden_states'][i]
      O2A_opinion_hidden_states=result_dict['O2A_opinion_hidden_states'][i]
      final_hidden_states=self.matching(O2A_aspect_hidden_states,O2A_opinion_hidden_states,O2A_aspect_terms,O2A_opinion_terms,calc_type='all',direct='O2A',mode='only-cls')
      O2A_logits=[]
      for idx in range(len(final_hidden_states)):
        row=final_hidden_states[idx]
        if torch.sum(row).item()==0 or idx in ignore_index:
          O2A_logits.append([0]*2)
        else:
          O2A_logits.append(self.sent_ffnn_O2A(row).tolist())
      
      ##Final Decision (nhãn sentiment của một token sẽ là trung bình cộng của hai chiều)
      A2O_logits=torch.tensor(A2O_logits)
      O2A_logits=torch.tensor(O2A_logits)
      final_logits=0.5*(A2O_logits+O2A_logits)
      
      ##Nếu ở chế độ train tính loss cho việc dán nhãn phân loại sentiment
      ##Trích xuất ra nhãn và logits tương ứng với những dòng logits toàn 0 để tránh việc tính toán loss bị sai
      if model_mode=='train':
        temp_final_logits=[]
        temp_sentiments=[]
        ##Trích xuất ra đúng những nhãn và logits của những tokens có logits khác không
        for inde in range(len(final_logits)):
          token_logits=final_logits[inde]
          if torch.sum(token_logits).item()==0:
            continue
          temp_final_logits.append(token_logits.tolist())
          temp_sentiments.append(result_dict['sentiment_labels_list'][i][inde])

        if self.args.ifgpu==True:
          temp_final_logits=torch.tensor(temp_final_logits).cuda()
          temp_sentiments=torch.tensor(temp_sentiments).cuda()
          weight = torch.tensor([1,2]).float().cuda()
        else:
          temp_final_logits=torch.tensor(temp_final_logits)
          temp_sentiments=torch.tensor(temp_sentiments)
          weight = torch.tensor([1,2]).float()

        ###Techer Forcing cho trường hợp temp_sentiments hoàn toàn chứa trù 1

        if torch.all(temp_sentiments == -1).item() == True:

          # ##Creating new aspect terms list
          # asp_answer=torch.tensor(batch_dict['aspect_answers'][i])
          # asp_one_index=(asp_answer == 1).nonzero(as_tuple=True)[0]
          # asp_two_index=(asp_answer == 2).nonzero(as_tuple=True)[0]
          # new_aspect_term=self._create_new_term_list(asp_one_index,asp_two_index)
          asp_answer=torch.tensor(batch_dict['aspect_answers'][i])
          asp_one_index=(asp_answer==1).nonzero(as_tuple=True)[0]
          new_aspect_term=asp_one_index.tolist()


          ##Creating new opinion terms list
          opi_answer=torch.tensor(batch_dict['opinion_answers'][i])
          opi_one_index=(opi_answer == 1).nonzero(as_tuple=True)[0]
          opi_two_index=(opi_answer == 2).nonzero(as_tuple=True)[0]
          new_opinion_term=self._create_new_term_list(opi_one_index,opi_two_index,ignore_index)

          ##A2O
          A2O_aspect_hidden_states=result_dict['A2O_aspect_hidden_states'][i]
          A2O_opinion_hidden_states=result_dict['A2O_opinion_hidden_states'][i]
          final_hidden_states=self.matching(A2O_aspect_hidden_states,A2O_opinion_hidden_states,new_aspect_term,new_opinion_term,calc_type='all',direct='A2O',mode='only-cls')
          A2O_logits=[]
          for idx in range(len(final_hidden_states)):
            row=final_hidden_states[idx]
            if torch.sum(row).item()==0 or idx in ignore_index:
              A2O_logits.append([0]*2)
            else:
              A2O_logits.append(self.sent_ffnn_A2O(row).tolist())

          ##O2A
          O2A_aspect_hidden_states=result_dict['O2A_aspect_hidden_states'][i]
          O2A_opinion_hidden_states=result_dict['O2A_opinion_hidden_states'][i]
          final_hidden_states=self.matching(O2A_aspect_hidden_states,O2A_opinion_hidden_states,new_aspect_term,new_opinion_term,calc_type='all',direct='O2A',mode='only-cls')
          O2A_logits=[]
          for idx in range(len(final_hidden_states)):
            row=final_hidden_states[idx]
            if torch.sum(row).item()==0 or idx in ignore_index:
              O2A_logits.append([0]*2)
            else:
              O2A_logits.append(self.sent_ffnn_O2A(row).tolist())

          ##Final Decision (nhãn sentiment của một token sẽ là trung bình cộng của hai chiều)
          A2O_logits=torch.tensor(A2O_logits)
          O2A_logits=torch.tensor(O2A_logits)
          final_logits=0.5*(A2O_logits+O2A_logits)


          temp_final_logits=[]
          temp_sentiments=[]
          ##Trích xuất ra đúng những nhãn và logits của những tokens có logits khác không
          for inde in range(len(final_logits)):
            token_logits=final_logits[inde]
            if torch.sum(token_logits).item()==0:
              continue
            temp_final_logits.append(token_logits.tolist())
            temp_sentiments.append(result_dict['sentiment_labels_list'][i][inde])

          if self.args.ifgpu==True:
            temp_final_logits=torch.tensor(temp_final_logits).cuda()
            temp_sentiments=torch.tensor(temp_sentiments).cuda()
          else:
            temp_final_logits=torch.tensor(temp_final_logits)
            temp_sentiments=torch.tensor(temp_sentiments)
        
        lossS+=F.cross_entropy(temp_final_logits,temp_sentiments,weight=weight,ignore_index=-1)
        # lossS=(1/self.args.batch_size)*lossS
      
      '''if model_mode=='train':
        pred=[]
        y_true=[]
        ##Trích xuất ra đúng những nhãn và logits của những tokens có logits khác không
        for inde in range(len(final_logits)):
          token_logits=final_logits[inde]
          if torch.sum(token_logits).item()==0:
            pred.append(-1)
          else:
            label_prob=F.softmax(token_logits,dim=-1)
            pred_label=torch.argmax(label_prob,dim=-1).item()
            pred.append(pred_label)
          y_true.append(result_dict['sentiment_labels_list'][i][inde])
        ##Nhãn dự đoán của toàn batch và nhãn đúng
        pred_list.append(pred)
        y_true_list.append(y_true)
        lossS=sentiment_loss(pred_list,y_true_list,ignore_index=-1)'''
      
      ##Getting label of aspect tokens:
      labels=[-1]*len(final_logits)
      for inde in range(len(final_logits)):
        token=final_logits[inde]
        if torch.sum(token).item()==0:
          continue
        max_index=torch.argmax(token).item()
        labels[inde]=max_index

      if 'deberta' in self.args.model_type:
        labels=self.filtered_sentiments(labels,batch_dict['texts'][i],self._tokenizer)

      predicts_list.append(labels)
      #aspects_list=self.filterOutput(result_dict['A2O_aspects_list'][i],result_dict['O2A_aspects_list'][i],batch_dict['texts_ids'][i],batch_dict['texts'][i],self._tokenizer,ignore_index=batch_dict['ignore_indexes'][i])
      aspects_list=list(set(result_dict['A2O_aspects_list'][i])&set(result_dict['O2A_aspects_list'][i]))
      if aspects_list==[]:
        aspects_list=list(set(result_dict['A2O_aspects_list'][i])|set(result_dict['O2A_aspects_list'][i]))
      opinions_list=self.filterOutput(result_dict['A2O_opinions_list'][i],result_dict['O2A_opinions_list'][i],batch_dict['texts_ids'][i],batch_dict['texts'][i],self._tokenizer,ignore_index=batch_dict['ignore_indexes'][i],mode='threshold')
      _aspects_list.append(aspects_list)
      _opinions_list.append(opinions_list)
    if model_mode=='train':
      lossS=(1/self.args.batch_size)*lossS
    return _aspects_list,_opinions_list,predicts_list,result_dict['lossA'],result_dict['lossO'],lossS
  
  def matching(self,aspect_hidden_states,opinion_hidden_states,aspect_terms,opinion_terms,calc_type='all',direct=None,mode='only-cls'):
    '''
      Hàm hỗ trợ Module kết nối hidden_state của aspect và opinion term tương ứng dựa trên điểm attention
      Mỗi apect term sẽ được kết nối với opinion terms có điểm attention cao nhất
    '''
    if mode=='only-cls':

      #Thử nghiệm số 1: chỉ sử dụng CLS như aspect hidden state
      opi_index=[idx for opinions in opinion_terms for idx in opinions]
      final_hidden_states=torch.zeros(opinion_hidden_states.size(0),self.args.hidden_size*2)
      for idx in opi_index:
        final_hidden_states[idx]=torch.cat((aspect_hidden_states,opinion_hidden_states[idx]),dim=-1)

    ##Những trường hợp khác    
    else:
      if direct=='A2O':
        if self.args.T%2!=0:
          hidden_states=opinion_hidden_states
        else:
          hidden_states=aspect_hidden_states
      elif direct=='O2A':
        if self.args.T%2!=0:
          hidden_states=aspect_hidden_states
        else:
          hidden_states=opinion_hidden_states
      attention_matrix,asp_index=self.calculate_attention(hidden_states,aspect_terms,opinion_terms,calc_type='all')
      max_vals,max_inds=torch.max(attention_matrix,dim=-1)
      final_hidden_states=torch.zeros(hidden_states.size(0),self.args.hidden_size*2)
      for idx in asp_index:
        final_hidden_states[idx]=torch.cat((aspect_hidden_states[idx],opinion_hidden_states[max_inds[idx].item()]),dim=-1)
    
    ##Kiểm tra xem có gpu không
    if self.args.ifgpu==True:
      return final_hidden_states.cuda()
    else:
      return final_hidden_states

  def calculate_attention(self,hidden_states,aspect_terms,opinion_terms,calc_type='all'):
    '''
      Hàm hỗ trợ module tính điểm attention the công thức do bài báo cung cấp.
      calc_type là phương pháp sẽ tính:
        + Với 'all' tất cả các aspect token dù là nhãn 1 hay 2 đều được xem là một aspect, opinion riêng biệt để tìm
        tương đồng với nhau.
        + Todo: sum hoặc averaged: sẽ tính tổng hoặc trung bình cộng hidden_states của các token nằm trong
        cùng một nhãn aspect hay opinion.
    '''
    n=hidden_states.size(1)
    score=torch.zeros(n,n)
    A=torch.zeros(n,n)
    ##Check if calc_type is all,we treat label 1 or 2 as the same
    if calc_type=='all':
      asp_index=[idx for aspects in aspect_terms for idx in aspects]
      opi_index=[idx for opinions in opinion_terms for idx in opinions]
      ##Calculate score first
      for idx in asp_index:
        for idy in opi_index:
          if idx!=idy:
            score[idx,idy]=torch.matmul(hidden_states[idx].t(),hidden_states[idy])/100
      ##Calculate attention score
      for idx in asp_index:
        for idy in opi_index:
          if idx!=idy:
            A[idx,idy]=torch.exp(score[idx,idy])/torch.exp(torch.sum(score[idx]))
    return A,asp_index

  def filterOutput(self,first_output,second_output,text_ids,text,_tokenizer=None,ignore_index=[],mode='merge'):
    '''
    Hàm để xử lý dữ liệu đầu ra cuối cùng:
      + Sau RoeFlipped Module ta hiện đang có hai danh sách aspect và hai danh sách opinion.
      + Hàm này hỗ trợ xóa bớt trùng ở cả hai hoặc ghép các aspect và opinion có sự trùng lắp về vị trí
       lại với nhau. Ví dụ: [3->5] với [4->7] sẽ được ghép thành [3->7]
    '''
    filtered_output=set()
    ##Adding first output to the filters
    for output in first_output:
      filtered_output.add((output[0],output[-1]))
    
    ##Adding second output to the filters
    for output in second_output:
      filtered_output.add((output[0],output[-1]))
    
    filtered_output=sorted(list(filtered_output),key=lambda x:(x[0],x[1]))
    for i in range(len(filtered_output)):
      filtered_output[i]=list(filtered_output[i])

    if mode=='merge':
      ##Filtered
      remove_ind=[]
      idx=0
      idy=1
      while idx<len(filtered_output)-1 and idy<len(filtered_output):
        if filtered_output[idx][0]==filtered_output[idy][0]:
          remove_ind.append(idx)
          idx+=1
          idy+=1
          continue
        elif filtered_output[idy][0]>filtered_output[idx][0]:
          if filtered_output[idy][0]<filtered_output[idx][1]:
            if filtered_output[idy][1]==filtered_output[idx][1]:
              remove_ind.append(idy)
              idy+=1
              continue
            else:
              filtered_output[idx][1]=filtered_output[idx][1]
              remove_ind.append(idy)
              idy+=1
              continue
          elif filtered_output[idy][0]==filtered_output[idx][1]:
            filtered_output[idx][1]=filtered_output[idx][1]
            remove_ind.append(idy)
            idy+=1
            continue
          else:
            idx+=1
            idy+=1
            continue

      ##Remove index:
      remove_ind=sorted(remove_ind)
      for ind in remove_ind[::-1]:
        filtered_output.pop(ind)


    elif mode=='threshold':
      pass

    
    if 'deberta' in self.args.model_type:
      index=torch.tensor(ignore_index)
      ignore_index=(index == -1).nonzero(as_tuple=True)[0]
      temp_text=[token.lower() for token in text]
      for i in range(len(filtered_output)):
        output=filtered_output[i]
        text_ids_index=text_ids[output[0]:output[1]+1]
        j=output[1]+1
        while j<len(text_ids) and j in ignore_index:
          text_ids_index.append(text_ids[j])
          j+=1
        result_text=_tokenizer.decode(text_ids_index,clean_up_tokenization_spaces=False).split()
        all_match=self.find_sub_list(result_text,temp_text)
        min_match=0
        match_distance=abs(all_match[0][0]-output[0])
        for a in range(1,len(all_match)):
          match=all_match[a]
          if abs(match[0]-output[0])<match_distance:
            min_match=a
            match_distance=abs(match[0]-output[0])
        filtered_output[i]=list(all_match[min_match])
    
    return filtered_output

  def find_sub_list(self,sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

  def filtered_sentiments(self,labels,text,_tokenizer=None):
    i=0
    j=0
    sentiments=[]
    temp_text=[token.lower() for token in text]
    while i<len(labels) and j<len(temp_text):
      ids=_tokenizer.encode(temp_text[j],add_special_tokens=False)
      sentiments.append(labels[i])
      i+=len(ids)
      j+=1
    return sentiments

  def _create_new_term_list(self,one_index,two_index,ignore_index):
    full_result=[]
    for idx in one_index:
      result=[]
      idx=idx.item()
      result.append(idx)
      idy=idx+1
      while idy in two_index and idy in ignore_index:
        if idy in two_index and idy not in ignore_index:
          result.append(idy)
        idy+=1
      full_result.append(result)
    return full_result



#Grab everything in to one model:
class RFMRC(nn.Module):
  ##Gom mọi thứ vào một model duy nhất
  def __init__(self,args):
    super(RFMRC,self).__init__()
    self.args=args

    ##RoleFlipped Module
    self._RF_Module=RoleFlippedModule(args)

    ##Matching Module
    self._Matching_Module=MatchingModule(args)

  def forward(self,batch_dict,model_mode='train'):
    result_dict=self._RF_Module(batch_dict,model_mode)
    return self._Matching_Module(result_dict,batch_dict,model_mode)