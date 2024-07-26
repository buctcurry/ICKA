import os
from torchvision import transforms
# from my_bert.tokenization import BertTokenizer
from local_transformers.adapter_transformers.models.roberta_ner import RobertaConfig, RobertaTokenizer, RobertaModel
# from local_transformers.adapter_transformers.models.bert_ner import BertConfig, BertTokenizer, BertModel
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.optimization import BertAdam, warmup_linear
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


import gc

import torch
from torch import nn
import torch.nn.functional as F
# from my_bert.gate_cl_modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, MTCCMBertForMMTokenClassificationCRF)
from Cross_Modal_Interaction_Module import MTCCMBertForMMTokenClassificationCRF
from resnet.resnet_utils import myResnet
import resnet.resnet as resnet
import argparse
from PIL import Image
import pickle
import transformers
from seqeval.metrics import classification_report
from ner_evaluate import evaluate
from my_bert.gate_cl_modeling import (CONFIG_NAME, WEIGHTS_NAME)
import json
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import cv2
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []

    print("The number of samples: %s", str(len(data)))
    return data

def mmreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename,encoding='utf-8')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    imgid = ''
    a = 0
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1] + '.jpg'
            continue
        if line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) > 0:
        data.append((sentence, label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: {}".format(len(data)))
    print("The number of images: " + str(len(imgs)))
    return data, imgs, auxlabels

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    try:
        image = transform(image)
    except:
        image = image.resize((224,224))
        image = transform(image)
    return image



class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None, clip_feature=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        # Please note that the auxlabel is just kept in order not to modify the original code
        self.auxlabel = auxlabel
        self.clip_feature = clip_feature

class  MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, img_feat, output_mask, label_id, 
                 auxlabel_id, 
                 ori_input_ids, ori_input_mask, ori_segment_ids, offset, clip_feature = None, rela_score = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.output_mask = output_mask
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id
        self.ori_input_ids = ori_input_ids
        self.ori_input_mask = ori_input_mask
        self.ori_segment_ids = ori_segment_ids
        self.clip_feature = clip_feature
        self.offset = offset
        self.rela_score = rela_score

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def _read_mmtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return mmreadfile(input_file)


class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "valid.txt"))
        return self._create_examples(data, imgs, auxlabels, data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, data_dir, "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]", '<s>', '</s>']

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "[CLS]", "[SEP]", '<s>', '</s>']

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, imgs, auxlabels, data_dir, set_type):
        with open (os.path.join(data_dir, 'Clip/' + set_type + "_features.pkl"), 'rb') as f:
            clip = pickle.load(f)

        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            clip_feature = clip[img_id.split(".")[0]]['text_features']
            examples.append(
                MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel, clip_feature=clip_feature))
        return examples


# input中的word被分割成sub-words，例如 SHOWS 变成 SH ##OW ##S
# 同理，其标签也会被重新分配，sub-words中的第一个是原标签，其他为X，例如SH ##OW ##S的标签为 O X X
def convert_mm_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, crop_size,
                                    path_img):
    
    # rela_score_mp = {}
    # with open('/home/yuanminghui/dataset/twitter2015/Tara_Divergence_Estimator/output_twitter_2015_test.json', 'r', encoding='utf-8') as f:
    #     test_rela_score_mp = json.load(f)
    #     rela_score_mp.update(test_rela_score_mp)

    # with open('/home/yuanminghui/dataset/twitter2015/Tara_Divergence_Estimator/output_twitter_2015_train.json', 'r', encoding='utf-8') as f:
    #     train_rela_score_mp = json.load(f)
    #     rela_score_mp.update(train_rela_score_mp)

    # with open('/home/yuanminghui/dataset/twitter2015/Tara_Divergence_Estimator/output_twitter_2015_valid.json', 'r', encoding='utf-8') as f:
    #     valid_rela_score_mp = json.load(f)
    #     rela_score_mp.update(valid_rela_score_mp)



    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}


    features = []
    count = 0
    
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
# for roberta
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
# for bert
    # bos_token = tokenizer.cls_token
    # eos_token = tokenizer.sep_token


#   总输入 = bos + prompt_text  + 'Text is' + bos + ori_input + eos
#   ori_tokens = bos + ori_input + eos
    prompt_text = 'Image is <mask> ' + \
                    'Bridge between Image and the Text is <mask> '

# =======for bert
    # prompt_text = 'Image is mask ' + \
    #                 'Bridge between Image and the Text is mask '

#   总输入 = bos + prompt_text + eos + ori_input-bos + eos
#   ori_tokens = bos + ori_input + eos
    # prompt_text = '<mask> ' + \
    #                   '<mask> '
    
    max_input_length = max_seq_length + len(prompt_text.split(" ")) + 30
    # f = open("kk.txt", 'w', encoding='utf-8')
    for (ex_index, example) in enumerate(examples):

        all_input = bos_token + " " +  prompt_text  + eos_token +  ' Text is '
    

        all_input_textlist = all_input.split(' ')
        
        clip_text_feature = example.clip_feature

        prompt_tokens = []
        
        for i, word in enumerate(all_input_textlist):
            token = tokenizer.tokenize(word)
            prompt_tokens.extend(token)


        #真正的输入也要记录 
        # labels和auxlabels只用来针对ori_input的标签
        ori_input = bos_token + " " + example.text_a + " " + eos_token
        ori_input_textlist = ori_input.split(" ")
        labellist = [label_map[bos_token]] + example.label + [label_map[eos_token]]
        auxlabellist = [label_map[bos_token]] + example.auxlabel + [label_map[eos_token]]
  
        labels = []
        auxlabels = []
        ori_tokens = []
        for i, word in enumerate(ori_input_textlist):
            if word == eos_token or word == bos_token:
                token = tokenizer.tokenize(word)
                ori_tokens.extend(token)
                labels.append(word)
                auxlabels.append(word)            
            else:
                # ['@', 'BBC', '##W', '##or', '##ld']
                token = tokenizer.tokenize(word)
                ori_tokens.extend(token)
                label_1 = labellist[i]
                auxlabel_1 = auxlabellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        auxlabels.append(auxlabel_1)
                    else:
                        labels.append("X")
                        auxlabels.append("X")

        if len(ori_tokens) >= max_seq_length - 1:
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        

        # 只用来针对ori_input的标签,
        # label和auxlabel

        ori_segment_ids = [0]*len(ori_tokens)
        label_ids = []
        auxlabel_ids = []
        for i, token in enumerate(labels):
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        # output_mask是用来在最后一层crf做mask的，他应该和label_ids始终一样的形状。同时也用来做评价时的循环
        output_mask = [1] * len(label_ids)
        
        ori_input_ids = tokenizer.convert_tokens_to_ids(ori_tokens)
        ori_input_mask = [1] * len(ori_input_ids)
        added_input_mask = [1] * (len(ori_input_ids) + 49)  # 1 or 49 is for encoding regional image representations
        
        while len(ori_input_ids) < max_seq_length:
            ori_input_ids.append(0)
            ori_input_mask.append(0)
            ori_segment_ids.append(0)
            added_input_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            auxlabel_ids.append(0)
            output_mask.append(0)
        # with open('t.txt', 'a', encoding='utf-8') as f:
        #     for i in range(len(tokenizer.convert_ids_to_tokens(ori_input_ids))):
        #         f.write(tokenizer.convert_ids_to_tokens(ori_input_ids)[i] + "\t" + str(label_ids[i]) + '\t' + str(ori_input_mask[i]) + '\n')
        #     f.write('\n')
        
        # print(tokenizer.convert_ids_to_tokens(ori_input_ids))
        # print(label_ids)
        # print(ori_input_mask)


        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        offset = len(prompt_ids)

  
        input_ids = prompt_ids + ori_input_ids
        input_tokens = prompt_tokens + ori_tokens

# ==============查看输出========================
        # embedding_output_1 = input_tokens[:3]
        # embedding_output_2 = input_tokens[4:11]
        # embedding_output_3 = input_tokens[12:]
        # input_tokens = embedding_output_1 + ['kkk'] + embedding_output_2 + ['kkkk'] + embedding_output_3
        # for i, tokens in enumerate(input_tokens):
        #     f.write(tokens +  '\n')
        # f.write('\n')
        # for i, tokens in enumerate(labels):
        #     f.write( input_tokens[offset+i]+ '\t' + str(tokens) +  '\n')
        # f.write('\n')        
# ==============查看输出========================


        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(prompt_ids) 
  
        while len(input_ids) < max_input_length:
            input_ids.append(0)
            input_mask.append(0)
        while len(segment_ids) < max_input_length:    
            segment_ids.append(1)
        

        assert len(input_ids) == max_input_length
        assert len(input_mask) == max_input_length
        assert len(segment_ids) == max_input_length
        assert len(ori_input_ids) == max_seq_length
        assert len(ori_input_mask) == max_seq_length
        assert len(ori_segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length
        assert len(label_ids) == len(output_mask)

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, transform)
        except:
            count += 1
            # print('image has problem!')
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)

        # rela_score = rela_score_mp[image_name]
        # print(image.shape)    torch.Size([3, 224, 224])

        # print(ntokens)
        # print(auxlabels)
        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
        #     logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))
        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                            segment_ids=segment_ids, img_feat=image, output_mask = output_mask,label_id=label_ids,
                            auxlabel_id=auxlabel_ids, ori_input_ids=ori_input_ids, ori_input_mask=ori_input_mask,
                            ori_segment_ids = ori_segment_ids,
                            offset=offset, clip_feature=clip_text_feature, rela_score = 1))
    print('the number of problematic samples: ' + str(count))
    # f.close()
    return features


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--negative_rate",
                        default=16,
                        type=int,
                        help="the negative samples rate")

    parser.add_argument('--lamb',
                        default=0.62,
                        type=float)

    parser.add_argument('--temp',
                        type=float,
                        default=0.179,
                        help="parameter for CL training")

    parser.add_argument('--temp_lamb',
                        type=float,
                        default=0.7,
                        help="parameter for CL training")

    parser.add_argument("--data_dir",
                        default='D:/PycharmProjects/Dataset/twitter15_data\data/twitter2015/',
                        type=str,

                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='twitter2015',
                        type=str,

                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='./output_result_2015_full/roberta_res/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--num_train_epochs",
                        default=25.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=19260817,
                        help="random seed for initialization")
    
    parser.add_argument('--CrossAtt_maskRate',
                        type=float,
                        default=0.5,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=5,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--mm_model', default='MTCCMBert', help='model name')  # 'MTCCMBert', 'NMMTCCMBert'
    parser.add_argument('--layer_num1', type=int, default=5, help='number of txt2img layer')
    parser.add_argument('--layer_num2', type=int, default=2, help='number of img2txt layer')
    parser.add_argument('--layer_num3', type=int, default=2, help='number of txt2txt layer')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='resnet', help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='twitter2017_images/', help='path to images')
    # parser.add_argument('--mm_model', default='TomBert', help='model name') #
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    
    if args.task_name == "twitter2017":
        args.path_image = "/home/yuanminghui/dataset/twitter2017/images/"
    elif args.task_name == "twitter2015":
        args.path_image = "D:/PycharmProjects/Dataset/twitter15_data/data/twitter2015_images/"
    


    return args


def train_and_dev():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    processor = MNERProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    # print(train_examples[0].guid)
    # print(train_examples[0].text_a)
    # print(train_examples[0].text_b)
    # print(train_examples[0].img_id)
    # print(train_examples[0].label)
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1
    temp = args.temp
    temp_lamb = args.temp_lamb
    lamb = args.lamb
    negative_rate = args.negative_rate

    print("local_rank is %d" % (args.local_rank))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    

    LAST_ENCODER_PATH = "encoder/roberta_large/"
    last_encoder_tokenizer = RobertaTokenizer.from_pretrained(LAST_ENCODER_PATH, do_lower_case=args.do_lower_case)
    last_encoder_config = RobertaConfig.from_pretrained(LAST_ENCODER_PATH + "config.json")
    last_encoder = RobertaModel.from_pretrained(LAST_ENCODER_PATH, config=last_encoder_config)
    last_encoder.resize_token_embeddings(len(last_encoder_tokenizer))
    last_encoder.config.type_vocab_size = 2
    last_encoder.embeddings.token_type_embeddings = nn.Embedding(2, last_encoder.config.hidden_size)
    
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    EMBEDDING_PATH = "embedding_bert/roberta_large/"
    embedding_tokenizer = transformers.AutoTokenizer.from_pretrained(EMBEDDING_PATH, do_lower_case=args.do_lower_case)
    embedding_config = transformers.RobertaConfig.from_pretrained(EMBEDDING_PATH)
    embedding = transformers.RobertaModel.from_pretrained(EMBEDDING_PATH,embedding_config)


    # vit_model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)
    model = MTCCMBertForMMTokenClassificationCRF(embedding_config,
                                                embedding = embedding,
                                                last_encoder=last_encoder,
                                                # cache_dir=cache_dir, 
                                                layer_num1=args.layer_num1,
                                                layer_num2=args.layer_num2,
                                                layer_num3=args.layer_num3,
                                                num_labels=num_labels)

    train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, embedding_tokenizer, args.crop_size, args.path_image)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in train_features])
    all_output_mask = torch.tensor([f.output_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_clip_features = torch.stack([f.clip_feature for f in train_features])
    all_ori_input_ids = torch.tensor([f.ori_input_ids for f in train_features], dtype=torch.long)
    all_ori_input_mask = torch.tensor([f.ori_input_mask for f in train_features], dtype=torch.long)
    all_ori_segment_ids = torch.tensor([f.ori_segment_ids for f in train_features], dtype=torch.long)
    all_offset = torch.tensor([f.offset for f in train_features], dtype=torch.long)
    all_rela_scores = torch.tensor([f.rela_score for f in train_features])
    # =================train_data
    train_data = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_img_feats, \
                               all_output_mask, all_label_ids, all_clip_features, \
                               all_ori_input_ids, all_ori_input_mask, all_ori_segment_ids, all_added_input_mask, all_offset, all_rela_scores)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data) 
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    # =================train_data_over



    dev_eval_examples = processor.get_dev_examples(args.data_dir)
    dev_eval_features = convert_mm_examples_to_features(
        dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, embedding_tokenizer, args.crop_size, args.path_image)
    all_input_ids = torch.tensor([f.input_ids for f in dev_eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in dev_eval_features], dtype=torch.long)
    all_added_input_mask = torch.tensor([f.added_input_mask for f in dev_eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in dev_eval_features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in dev_eval_features])
    all_label_ids = torch.tensor([f.label_id for f in dev_eval_features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in dev_eval_features], dtype=torch.long)
    all_clip_features = torch.stack([f.clip_feature for f in dev_eval_features])
    all_ori_input_ids = torch.tensor([f.ori_input_ids for f in dev_eval_features], dtype=torch.long)
    all_ori_input_mask = torch.tensor([f.ori_input_mask for f in dev_eval_features], dtype=torch.long)
    all_ori_segment_ids = torch.tensor([f.ori_segment_ids for f in dev_eval_features], dtype=torch.long)
    all_offset = torch.tensor([f.offset for f in dev_eval_features], dtype=torch.long)
    all_rela_scores = torch.tensor([f.rela_score for f in dev_eval_features])
    # =================dev_data
    dev_eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_img_feats, \
                               all_output_mask, all_label_ids, all_clip_features, \
                               all_ori_input_ids, all_ori_input_mask, all_ori_segment_ids, all_added_input_mask, all_offset, all_rela_scores)
    dev_eval_sampler = SequentialSampler(dev_eval_data)
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=dev_eval_sampler, batch_size=args.eval_batch_size)
    # =================dev_data_over
    
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.01)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                         lr=args.learning_rate,
    #                         warmup=args.warmup_proportion,
    #                         t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_proportion*num_train_optimization_steps, num_training_steps=num_train_optimization_steps)
    
    output_model_file = os.path.join(args.output_dir, 'model.pth')
    output_resnet_file = os.path.join(args.output_dir, "pytorch_resnet.bin")

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    model.to(device)
    encoder.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
        encoder = DDP(encoder)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)


    max_dev_f1 = -1.0
    best_dev_epoch = 0
    global_step = 0
    print("***** Running training *****")
    for train_idx in range(int(args.num_train_epochs)):
        print("********** Epoch: " + str(train_idx) + " **********")
        print("  Num examples =  {}".format(len(train_examples)))
        print("  Batch size =  {}".format(args.train_batch_size))
        print("  Num steps = {}".format(num_train_optimization_steps))
        model.train()
        encoder.train()
        encoder.zero_grad()
        nb_tr_examples, nb_tr_steps = 0, 0
        total_loss = 0
        index = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, img_feats, all_output_mask, label_ids, clip_features, \
                ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, offsets, rela_scores = batch
            
            assert torch.all(offsets == offsets[0])

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)


            # imgs_f = [batch_size, 2048]
            # img_mean = [batch_size, 2048]
            # img_att = [batch_size, 2048, 7, 7]
            # zero_ratio = args.CrossAtt_maskRate
            # random_mask = torch.rand(input_ids.shape[0], 49).to(device) > zero_ratio

            neg_log_likelihood = model(input_ids, segment_ids, input_mask, \
                  ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, clip_features,
                 imgs_f, img_att, offsets, all_output_mask, rela_scores,
                 temp,temp_lamb,lamb,label_ids, negative_rate, mode='train')
     
            if n_gpu > 1:
                neg_log_likelihood = neg_log_likelihood.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(neg_log_likelihood)
            else:
                neg_log_likelihood.backward()

            total_loss += neg_log_likelihood.item()
            index += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                        args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        
        print("                         Train_LOSS =  {}".format(total_loss/index*args.gradient_accumulation_steps))
        model.eval()
        encoder.eval()

        print("***** Running Dev evaluation *****")
        print("  Num examples =  {}".format(len(dev_eval_examples)))
        print("  Batch size =  {}".format(args.eval_batch_size))
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        dev_total_loss = 0
        index = 0
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"
        for step, batch in enumerate(dev_eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, img_feats, all_output_mask, label_ids, clip_features, \
                ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, offsets, rela_scores = batch


            # zero_ratio = args.CrossAtt_maskRate
            # random_mask = torch.rand(input_ids.shape[0], 49).to(device) > zero_ratio

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids, loss = model(input_ids, segment_ids, input_mask, \
                  ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, clip_features,
                 imgs_f, img_att, offsets, all_output_mask, rela_scores, 
                 temp,temp_lamb,lamb,labels=label_ids, negative_rate=None, mode = 'dev')

            dev_total_loss += loss.item()
            index += 1
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            ori_input_mask = ori_input_mask.to('cpu').numpy()
            for i, mask in enumerate(all_output_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                for j, m in enumerate(mask):
                    # if j == 0:
                    #     continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>" \
                            and label_map[label_ids[i][j]] != "<s>" and label_map[label_ids[i][j]] != "[CLS]"\
                            and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)
        print("                         DEV_LOSS =  {}".format(dev_total_loss/index))
        report = classification_report(y_true, y_pred, digits=4)
        sentence_list = []
        dev_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "valid.txt"))
        for i in range(len(y_pred)):
            sentence = dev_data[i][0]
            sentence_list.append(sentence)
        reverse_label_map = {label: i for i, label in enumerate(label_list,1)}
        reverse_label_map['PAD'] = 0
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, y_true, sentence_list, reverse_label_map)
        
        print("p = {}, r = {}, f1 = {} ".format(p, r, f1))
        F_score_dev = f1
        if F_score_dev > max_dev_f1:
            
            print("***** Dev Eval results *****")
            print("\n" + report)
            print("Overall:p = {}, r = {}, f1 = {} ".format(p, r, f1))
            # Save a trained model and the associated configuration
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # encoder_to_save = encoder.module if hasattr(encoder,
            #                                             'module') else encoder  # Only save the model it-self

            label_map = {i: label for i, label in enumerate(label_list, 1)}
            model_config = {"bert_model": args.bert_model, "do_lower_case": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                            "output_dir":args.output_dir, "resnet_root":args.resnet_root,
                            "layer_num1":args.layer_num1, "layer_num2": args.layer_num2, 
                            "layer_num3":args.layer_num3, "path_image": args.path_image,
                            "label_map": label_map, "data_dir": args.data_dir}
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))

            
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, output_model_file, _use_new_zipfile_serialization=False)
            torch.save(encoder.state_dict(), output_resnet_file)

            path = {"LAST_ENCODER_PATH": LAST_ENCODER_PATH, "EMBEDDING_PATH":EMBEDDING_PATH}
            json.dump(path, open(os.path.join(args.output_dir, "path.json"), "w"))
            max_dev_f1 = F_score_dev

    print(max_dev_f1)   


def test(output_dir):
    output_args_file = os.path.join(output_dir, "model_config.json")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args_dict = vars(args)
    with open(output_args_file, 'rt') as f:
        args_dict.update(json.load(f))

    processor = MNERProcessor()

    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1


    device = torch.device("cuda")

    
    output_model_file = os.path.join(args.output_dir, 'model.pth')
    output_resnet_file = os.path.join(args.output_dir, "pytorch_resnet.bin")
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, False, device)
    encoder_state_dict = torch.load(output_resnet_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    path = json.load(open(os.path.join(args.output_dir, 'path.json'),'r'))
    LAST_ENCODER_PATH = path["LAST_ENCODER_PATH"]
    last_encoder_tokenizer = RobertaTokenizer.from_pretrained(LAST_ENCODER_PATH, do_lower_case=args.do_lower_case)
    last_encoder_config = RobertaConfig.from_pretrained(LAST_ENCODER_PATH + "config.json")
    last_encoder = RobertaModel.from_pretrained(LAST_ENCODER_PATH, config=last_encoder_config)
    last_encoder.resize_token_embeddings(len(last_encoder_tokenizer))
    last_encoder.config.type_vocab_size = 2
    last_encoder.embeddings.token_type_embeddings = nn.Embedding(2, last_encoder.config.hidden_size)

    EMBEDDING_PATH = path["EMBEDDING_PATH"]
    embedding_tokenizer = transformers.AutoTokenizer.from_pretrained(EMBEDDING_PATH, do_lower_case=args.do_lower_case)
    embedding_config = transformers.RobertaConfig.from_pretrained(EMBEDDING_PATH)
    embedding = transformers.RobertaModel.from_pretrained(EMBEDDING_PATH,embedding_config)

    model = MTCCMBertForMMTokenClassificationCRF(embedding_config,
                                                embedding = embedding,
                                                last_encoder=last_encoder,
                                                # cache_dir=cache_dir, 
                                                layer_num1=args.layer_num1,
                                                layer_num2=args.layer_num2,
                                                layer_num3=args.layer_num3,
                                                num_labels=num_labels)
    params = torch.load(output_model_file)['net']
    model.load_state_dict(params, False)
    model.to(device)


    test_eval_examples = processor.get_test_examples(args.data_dir)
    test_eval_features = convert_mm_examples_to_features(
        test_eval_examples, label_list, auxlabel_list, args.max_seq_length, embedding_tokenizer, 224, args.path_image)
    all_input_ids = torch.tensor([f.input_ids for f in test_eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_eval_features], dtype=torch.long)
    all_added_input_mask = torch.tensor([f.added_input_mask for f in test_eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_eval_features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in test_eval_features])
    all_output_mask = torch.tensor([f.output_mask for f in test_eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_eval_features], dtype=torch.long)
    all_clip_features = torch.stack([f.clip_feature for f in test_eval_features])
    all_ori_input_ids = torch.tensor([f.ori_input_ids for f in test_eval_features], dtype=torch.long)
    all_ori_input_mask = torch.tensor([f.ori_input_mask for f in test_eval_features], dtype=torch.long)
    all_ori_segment_ids = torch.tensor([f.ori_segment_ids for f in test_eval_features], dtype=torch.long)
    all_offset = torch.tensor([f.offset for f in test_eval_features], dtype=torch.long)
    # =================dev_data
    test_eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_img_feats, \
                               all_output_mask, all_label_ids, all_clip_features, \
                               all_ori_input_ids, all_ori_input_mask, all_ori_segment_ids, all_added_input_mask, all_offset)
    test_eval_sampler = SequentialSampler(test_eval_data)
    test_eval_dataloader = DataLoader(test_eval_data, sampler=test_eval_sampler, batch_size=4)


    model.eval()
    encoder.eval()

    print("***** Running TEST evaluation *****")
    print("  Num examples =  {}".format(len(test_eval_examples)))
    print("  Batch size =  {}".format(4))
    y_true = []
    y_pred = []
    y_true_idx = []
    y_pred_idx = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "PAD"
    for step, batch in enumerate(test_eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, img_feats, all_output_mask, label_ids, clip_features, \
            ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, offsets = batch

        # zero_ratio = args.CrossAtt_maskRate
        # random_mask = torch.rand(input_ids.shape[0], 49).to(device) > zero_ratio

        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feats)
            predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, \
                ori_input_ids, ori_input_mask, ori_segment_ids, added_input_mask, clip_features,
                imgs_f, img_att, offsets, all_output_mask, None,None,None,labels=None, 
                negative_rate=None, mode='test')

        logits = predicted_label_seq_ids
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, mask in enumerate(all_output_mask):
            temp_1 = []
            temp_2 = []
            tmp1_idx = []
            tmp2_idx = []
            for j, m in enumerate(mask):
                # if j == 0:
                #     continue
                if m:
                    if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>" \
                            and label_map[label_ids[i][j]] != "<s>" and label_map[label_ids[i][j]] != "[CLS]"\
                            and label_map[label_ids[i][j]] != "[SEP]":
                        temp_1.append(label_map[label_ids[i][j]])
                        tmp1_idx.append(label_ids[i][j])
                        temp_2.append(label_map[logits[i][j]])
                        tmp2_idx.append(logits[i][j])
                else:
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
            y_true_idx.append(tmp1_idx)
            y_pred_idx.append(tmp2_idx)
    report = classification_report(y_true, y_pred, digits=4)
    sentence_list = []
    dev_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
    for i in range(len(y_pred)):
        sentence = dev_data[i][0]
        sentence_list.append(sentence)
    reverse_label_map = {label: i for i, label in enumerate(label_list,1)}
    reverse_label_map['PAD'] = 0
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, y_true, sentence_list, reverse_label_map)
    
    print(report)
    print("p = {}, r = {}, f1 = {} ".format(p, r, f1))


if __name__ == "__main__":


    # =================================================train and dev =====================================================
    train_and_dev()


# =======================================TEST=======================================================================
    # test('output_result_2015_full/roberta_res')


# nohup python -u My_cross_attention.py > My_cross_attention_res.txt 2>&1 &
# python -u -m torch.distributed.launch --nproc_per_node=2 --rdzv_backend c10d --master_port=0 My_cross_attention.py > My_cross_attention_res.txt 2>&1 &