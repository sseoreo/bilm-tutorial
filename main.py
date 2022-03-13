import os

import random
import argparse
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utils import setup_exp
from dataset import create_dataset
from train import train, evaluate

parser = argparse.ArgumentParser('NN', add_help=False)
parser.add_argument('--freeze', default=False, action='store_true', help='not using finetune')

parser.add_argument('--dataset', type=str, default='ptb-context', choices = ['toy'], help='datasets.')
parser.add_argument('--model', default="bert-base-cased", type=str, choices = ['gpt2', 'bert-base-cased'], help="huggingface model") 
parser.add_argument('--pretrained', default=None, type=str, help="directory for pretrained weights")


parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode - no wandb')
parser.add_argument('--output_dir', default="results", type=str, help="directory for checkpoints")
parser.add_argument('--data', type=str, default='../revisit-nplm/data', help='location of the data corpus')

parser.add_argument('--epochs', default=3, type=int, help="total epochs")
parser.add_argument('--max_iter', default=10000, type=int, help="max iteration")

parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
parser.add_argument('--batch_length', default=256, type=int, help="batch length of input sequence")
parser.add_argument('--target_length', default=256, type=int, help="target length for evaluation")

parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--batch_size_valid', default=3, type=int, help="batch size for validation")


parser.add_argument('--warmup_step', default=400, type=int, help="warmpup step")
parser.add_argument('--interval_print', default=100, type=int, help="interval for logging-wandb,stdout")
parser.add_argument('--interval_valid', default=1000, type=int, help="interval for validation")

config = parser.parse_args()

if not config.debug:
    setup_exp.make_workspace(config, 'output_dir', 'model-dataset-lr-batch_size')

config.wandb = setup_exp.init_wandb(
                config, 
                os.path.basename(config.output_dir), 
                project_name='sent-to-vec', 
                debug=config.debug)

config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tokenizer = AutoTokenizer.from_pretrained(config.model)

print ("============= Loading for Data =============")
trainset, validset, vocab = create_dataset(config.dataset, tokenizer, config.batch_length, config.batch_size, split='train')
# validset = create_dataset(config.dataset, tokenizer, config.batch_length, config.batch_size_valid, split='validation')


if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token = tokenizer.pad_token_id
else: 
    config.pad_token = tokenizer.pad_token_id
    

model = AutoModelForSequenceClassification.from_pretrained(
                                        config.model,
                                        num_labels=vocab.size,
                                        output_attentions = False)
model.to(config.device)
# if config.model == 'gpt2':
#     device_map = {
#         0: [0, 1, ],
#         1: [2, 3, 4, 5,],
#         2: [6, 7, 8,],
#         3: [9, 10, 11]
#     }
#     model.parallelize(device_map)





params = list(model.named_parameters())

print(f'The {config.model} model has {len(params)} different named parameters.\n')

print('==== Embedding Layer ====\n')

if config.freeze:
    for n, p in params:
        if 'bert.encoder' in n: p.requires_grad = False

for p in params[0:5]:
    print("{:<55} {:>12} {:>10}".format(p[0], str(tuple(p[1].size())), "trainable" if p[1].requires_grad else 'freeze'), )

print('\n==== First Transformer ====\n')
    
for p in params[5:21]:
    print("{:<55} {:>12} {:>10}".format(p[0], str(tuple(p[1].size())), "trainable" if p[1].requires_grad else 'freeze'), )

print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12} {:>10}".format(p[0], str(tuple(p[1].size())), "trainable" if p[1].requires_grad else 'freeze'), )


optimizer = AdamW(model.parameters(),
                  lr = config.lr, # config.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # config.adam_epsilon  - default is 1e-8.
                  )


scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = config.warmup_step, # Default value in run_glue.py
                                            num_training_steps = config.max_iter)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


train(config, trainset, model, optimizer, scheduler, validset=validset)

# print("="*100)
# print('Testing...')
# testset = create_dataset(config.dataset, tokenizer, config.batch_length, config.batch_size, split='test')
# test_loss, test_ppl = evaluate(config, testset, model) 
# print(f"tloss:{test_loss} tppl: {test_ppl}")

# if not config.debug:
#     with open(os.path.join(config.output_dir, 'model.pt'), 'wb') as f:
#         torch.save({'model': model}, f)