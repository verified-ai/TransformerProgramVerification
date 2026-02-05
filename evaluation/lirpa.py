import argparse
import pickle
import importlib
import json
import os
from collections import defaultdict, namedtuple
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor, CrossEntropyWrapperMultiInput
from auto_LiRPA.perturbations import PerturbationLpNorm, PerturbationSynonym
from auto_LiRPA.utils import Flatten, MultiAverageMeter, scale_gradients
from torch.utils.tensorboard import SummaryWriter
from auto_LiRPA.eps_scheduler import LinearScheduler
from examples.language.Transformer.Transformer import Transformer
from examples.language.data_utils import get_batches
import time
from z3 import *

from src.models.programs import (
    TransformerProgramModel,
    argmax,
    gumbel_hard,
    gumbel_soft,
    softmax,
)
from src.utils import data_utils

# -------------AI suggestion for bound on hamming distance on outputs---------------

class HammingDistanceModel(nn.Module):
    def __init__(self, model, correct_indices):
        super().__init__()
        self.model = model
        self.correct_indices = correct_indices  # list of correct token indices for each position

    def forward(self, x):
        logits = self.model(x)  # shape: (batch, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)
        hamming = 0.0
        for i in range(logits.shape[1]):
            hamming += 1 - probs[:, i, self.correct_indices[i]]
        return hamming.mean()  # scalar: expected Hamming distance

def get_sorted_tokens(tokens): #todo make right
    # Sort the tokens numerically, keeping <s> and </s> in place
    # Assume tokens are like ["<s>", "3", "1", "2", "</s>"]
    numbers = []
    for t in tokens:
        if t in ["<s>", "</s>"]:
            numbers.append(-1 if t == "<s>" else 100)  # place <s> first, </s> last
        else:
            numbers.append(int(t))
    sorted_numbers = sorted(numbers)
    sorted_tokens = []
    for n in sorted_numbers:
        if n == -1:
            sorted_tokens.append("<s>")
        elif n == 100:
            sorted_tokens.append("</s>")
        else:
            sorted_tokens.append(str(n))
    return sorted_tokens

def lirpa_hamming_distance_bounds(model, x, correct_tokens, ptb, idx_w):

    # Map correct tokens to indices
    correct_indices = [idx_w[token] for token in correct_tokens]

    # Create the Hamming distance model
    hamming_model = HammingDistanceModel(model, correct_indices)

    # Set up LIRPA
    lirpa_model = BoundedModule(hamming_model, x)
    lirpa_model.visualize("lirpa_hamming_model_graph")

    bounded_x = BoundedTensor(x, ptb)

    for method in ['IBP', 'CROWN', 'alpha-CROWN']:
        start_time = time.time()
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
        end_time = time.time()
        print(f'{method} bounds on expected Hamming distance: lower={lb.item()}, upper={ub.item()}, time={end_time - start_time:.4f}s')

#---------------Load data and transformer program model----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_sample_fn(name):
    d = {
        "softmax": softmax,
        "gumbel_hard": gumbel_hard,
        "gumbel_soft": gumbel_soft,
    }
    if name not in d:
        raise NotImplementedError(name)
    return d[name]

def load_dataset(args):
    return data_utils.get_dataset(
        name=args["dataset"],
        vocab_size=args["vocab_size"],
        dataset_size=args["dataset_size"],
        min_length=args["min_length"],
        max_length=args["max_length"],
        seed=args["seed"],
        do_lower=args["do_lower"],
        replace_numbers=args["replace_numbers"],
        get_val=True,
        unique=args["unique"],
    )

def load_transformer_program(file_path):
    
    with open(Path(file_path) / "args.json", "r") as f:
        args = json.load(f)

    set_seed(args["seed"])
    (
        train,
        test,
        val,
        idx_w,
        w_idx,
        idx_t,
        t_idx,
        X_train,
        Y_train,
        X_test,
        Y_test,
        X_val,
        Y_val,
    ) = load_dataset(args)

    if args["d_var"] is None:
        d = max(len(idx_w), X_train.shape[-1])
    else:
        d = args["d_var"]
    init_emb = None
    if args["glove_embeddings"] and args["do_glove"]:
        emb = data_utils.get_glove_embeddings(idx_w, args["glove_embeddings"], dim=args["n_vars_cat"] * d)
        init_emb = torch.tensor(emb, dtype=torch.float32).T
    unembed_mask = None
    if args["unembed_mask"]:
        unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t])


    model = TransformerProgramModel(
        d_vocab=len(idx_w),
        d_vocab_out=len(idx_t),
        n_vars_cat=args["n_vars_cat"],
        n_vars_num=args["n_vars_num"],
        d_var=d,
        n_heads_cat=args["n_heads_cat"],
        n_heads_num=args["n_heads_num"],
        d_mlp=args["d_mlp"],
        n_cat_mlps=args["n_cat_mlps"],
        n_num_mlps=args["n_num_mlps"],
        mlp_vars_in=args["mlp_vars_in"],
        n_layers=args["n_layers"],
        n_ctx=X_train.shape[1],
        sample_fn=get_sample_fn(args["sample_fn"]),
        init_emb=init_emb,
        attention_type=args["attention_type"],
        rel_pos_bias=args["rel_pos_bias"],
        unembed_mask=unembed_mask,
        pool_outputs=args["pool_outputs"],
        one_hot_embed=args["one_hot_embed"],
        count_only=args["count_only"],
        selector_width=args["selector_width"],
    ).to(torch.device(args["device"]))
    model.load_state_dict(torch.load(Path(file_path) / "model.pt"))
    model.eval()

    # For LIRPA, use softmax instead of argmax to make the model differentiable
    """
    model.sample_fn = softmax
    for block in model.blocks:
        block.cat_mlp.sample_fn = softmax
        if block.num_mlp:
            block.num_mlp.sample_fn = softmax
        block.cat_attn.sample_fn = softmax
        if block.num_attn:
            block.num_attn.sample_fn = softmax
    """
    seq_len = X_train.shape[1]
    return model, idx_w, seq_len

#---------------Old LIRPA--------------------------

def lirpa_bounds(model: TransformerProgramModel, ptb, seq_len, idx_w):

    # Input: dummy tensor with correct shape for the model (batch=1, seq_len)
    # Use random valid token indices to avoid index out of range
    d_vocab = len(idx_w)
    x = torch.randint(0, d_vocab, (1, seq_len), dtype=torch.long)
    lirpa_model = BoundedModule(model, x)
    lirpa_model.visualize("lirpa_model_graph")

    bounded_x = BoundedTensor(x, ptb)

    for method in ['IBP', 'CROWN', 'alpha-CROWN']:
        start_time = time.time()
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
        end_time = time.time()
        print(f'{method} bounds: lower={lb.item()}, upper={ub.item()}, time={end_time - start_time:.4f}s')
    
    return True

def lirpa_lp_bounds(model, eps, seq_len, idx_w):
    ptb = PerturbationLpNorm(norm = float("inf"), eps=eps)
    lirpa_bounds(model, ptb, seq_len, idx_w)

def lirpa_synonym_bounds(model, budget, seq_len, synonyms, idx_w):
    with open("data/synonyms.json", "w") as f:
        json.dump(synonyms, f, indent=4)
    ptb = PerturbationSynonym(budget=budget)
    ptb.synonym = synonyms
    lirpa_bounds(model, ptb, seq_len, idx_w)

def evaluate_sorting_lirpa_on_transformer_program(input_length, model=None, idx_w=None, seq_len=None):

    #------lirpa----- how to include input length?

    if model is None:
        model, idx_w, seq_len = load_transformer_program("output/sort")
    synonyms = {x : (idx_w.tolist()) for x in idx_w}
    min, max = 0, input_length
    max_robust_budget = 0
    while min <= max:
        mid = (min + max) // 2
        print(f"Testing replacement budget {mid}")
        robust = lirpa_synonym_bounds(model, budget=mid, seq_len=seq_len, synonyms=synonyms, idx_w=idx_w)
        if robust:
            print("Robust")
            max_robust_budget = mid
            min = mid + 1
        else:
            print("Not Robust")
            max_robust_budget = mid - 1
            max = mid - 1
    print(f"Maximum lirpa synonym replacement budget for robustness: {max_robust_budget}")

#--------------LIRPA---------------------

def parse_args_lirpa():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--dir', type=str, default='model')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--data', type=str, default='sst', choices=['sst'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--legacy_loading', action='store_true', help='use a deprecated way of loading checkpoints for previously saved models')
    parser.add_argument('--auto_test', action='store_true')

    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--budget', type=int, default=6)
    parser.add_argument('--method', type=str, default=None,
                        choices=['IBP', 'IBP+backward', 'IBP+backward_train', 'forward', 'forward+backward'])

    parser.add_argument('--model', type=str, default='transformer',
                        choices=['transformer', 'lstm'])
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--num_epochs_all_nodes', type=int, default=20)
    parser.add_argument('--eps_start', type=int, default=1)
    parser.add_argument('--eps_length', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--min_word_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--oracle_batch_size', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_sent_length', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--drop_unk', action='store_true')
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--layer_norm', type=str, default='no_var',
                        choices=['standard', 'no', 'no_var'])
    parser.add_argument('--loss_fusion', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bound_opts_relu', type=str, default='zero-lb')

    args_lirpa = parser.parse_args()

    return args_lirpa

def step(args_lirpa, model, ptb, batch, model_loss, eps=1.0, train=False):
    model_bound = model.model_from_embeddings
    if train:
        model.train()
        model_bound.train()
        grad = torch.enable_grad()
        if args_lirpa.loss_fusion:
            model_loss.train()
    else:
        model.eval()
        model_bound.eval()
        grad = torch.no_grad()
    if args_lirpa.auto_test:
        grad = torch.enable_grad()

    with grad:
        ptb.set_eps(eps)
        ptb.set_train(train)
        embeddings_unbounded, mask, tokens, labels = model.get_input(batch)
        aux = (tokens, batch)
        if args_lirpa.robust and eps > 1e-9:
            embeddings = BoundedTensor(embeddings_unbounded, ptb)
        else:
            embeddings = embeddings_unbounded.detach().requires_grad_(True)

        robust = args_lirpa.robust and eps > 1e-6

        if train and robust and args_lirpa.loss_fusion:
            # loss_fusion loss
            if args_lirpa.method == 'IBP+backward_train':
                lb, ub = model_loss.compute_bounds(
                    x=(labels, embeddings, mask), aux=aux,
                    C=None, method='IBP+backward', bound_lower=False)
            else:
                raise NotImplementedError
            loss_robust = torch.log(ub).mean()
            loss = acc = acc_robust = -1 # unknown
        else:
            # regular loss
            logits = model_bound(embeddings, mask)
            loss = CrossEntropyLoss()(logits, labels)
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

            if robust:
                num_class = args_lirpa.num_classes
                c = torch.eye(num_class).type_as(embeddings)[labels].unsqueeze(1) - \
                    torch.eye(num_class).type_as(embeddings).unsqueeze(0)
                I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
                c = (c[I].view(embeddings.size(0), num_class - 1, num_class))
                if args_lirpa.method in ['IBP', 'IBP+backward', 'forward', 'forward+backward']:
                    lb, ub = model_bound.compute_bounds(aux=aux, C=c, method=args_lirpa.method, bound_upper=False)
                elif args_lirpa.method == 'IBP+backward_train':
                    # CROWN-IBP
                    if 1 - eps > 1e-4:
                        lb, ub = model_bound.compute_bounds(aux=aux, C=c, method='IBP+backward', bound_upper=False)
                        ilb, iub = model_bound.compute_bounds(aux=aux, C=c, method='IBP', reuse_ibp=True)
                        lb = eps * ilb + (1 - eps) * lb
                    else:
                        lb, ub = model_bound.compute_bounds(aux=aux, C=c, method='IBP')
                else:
                    raise NotImplementedError
                lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
                fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
                loss_robust = robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
                acc_robust = 1 - torch.mean((lb < 0).any(dim=1).float())
            else:
                acc_robust, loss_robust = acc, loss

    if train or args_lirpa.auto_test:
        loss_robust.backward()
        grad_embed = torch.autograd.grad(
            embeddings_unbounded, model.word_embeddings.weight,
            grad_outputs=embeddings.grad)[0]
        if model.word_embeddings.weight.grad is None:
            model.word_embeddings.weight.grad = grad_embed
        else:
            model.word_embeddings.weight.grad += grad_embed

    if args_lirpa.auto_test:
        print('Saving results for automated tests.')
        print(f'acc={acc}, loss={loss}, robust_acc={acc_robust}, robust_loss={loss_robust}')
        print('gradients:')
        print(grad_embed)
        with open('res_test.pkl', 'wb') as file:
            pickle.dump((
                float(acc), float(loss), float(acc_robust), float(loss_robust),
                grad_embed.detach().numpy()), file)

    return acc, loss, acc_robust, loss_robust

def train(args_lirpa, epoch, batches, type, optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer):
    meter = MultiAverageMeter()
    assert(optimizer is not None)
    train = type == 'train'
    if args_lirpa.robust:
        eps_scheduler.set_epoch_length(len(batches))
        if train:
            eps_scheduler.train()
            eps_scheduler.step_epoch()
        else:
            eps_scheduler.eval()
    for i, batch in enumerate(batches):
        if args_lirpa.robust:
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
        else:
            eps = 0
        acc, loss, acc_robust, loss_robust = step(
            args_lirpa, model, ptb, batch, model_loss, eps=eps, train=train)
        meter.update('acc', acc, len(batch))
        meter.update('loss', loss, len(batch))
        meter.update('acc_rob', acc_robust, len(batch))
        meter.update('loss_rob', loss_robust, len(batch))
        if train:
            if (i + 1) % args_lirpa.gradient_accumulation_steps == 0 or (i + 1) == len(batches):
                scale_gradients(optimizer, i % args_lirpa.gradient_accumulation_steps + 1, args_lirpa.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            writer.add_scalar('loss_train_{}'.format(epoch), meter.avg('loss'), i + 1)
            writer.add_scalar('loss_robust_train_{}'.format(epoch), meter.avg('loss_rob'), i + 1)
            writer.add_scalar('acc_train_{}'.format(epoch), meter.avg('acc'), i + 1)
            writer.add_scalar('acc_robust_train_{}'.format(epoch), meter.avg('acc_rob'), i + 1)
        if (i + 1) % args_lirpa.log_interval == 0 or (i + 1) == len(batches):
            print('Epoch {}, {} step {}/{}: eps {:.5f}, {}'.format(
                epoch, type, i + 1, len(batches), eps, meter))
            if lr_scheduler is not None:
                print('lr {}'.format(lr_scheduler.get_lr()))
    writer.add_scalar('loss/{}'.format(type), meter.avg('loss'), epoch)
    writer.add_scalar('loss_robust/{}'.format(type), meter.avg('loss_rob'), epoch)
    writer.add_scalar('acc/{}'.format(type), meter.avg('acc'), epoch)
    writer.add_scalar('acc_robust/{}'.format(type), meter.avg('acc_rob'), epoch)

    if train:
        if args_lirpa.loss_fusion:
            state_dict_loss = model_loss.state_dict()
            state_dict = {}
            for name in state_dict_loss:
                assert(name.startswith('model.'))
                state_dict[name[6:]] = state_dict_loss[name]
            model_ori.load_state_dict(state_dict)
            model_bound = BoundedModule(
                model_ori, (dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=args_lirpa.device)
            model.model_from_embeddings = model_bound
        model.save(epoch)

    return meter.avg('acc_rob')

def init_bounded_transformer(file_path, generate_synonyms, args_lirpa):

    with open(Path(file_path) / "args.json", "r") as f:
        args = json.load(f)
    
    (
        train_data,
        test_data,
        val,
        idx_w,
        w_idx,
        idx_t,
        t_idx,
        X_train,
        Y_train,
        X_test,
        Y_test,
        X_val,
        Y_val,
    ) = load_dataset(args)

    #train_data = train_data.rename(columns={'sent': 'sentence'})
    #test_data = test_data.rename(columns={'sent': 'sentence'})
    # Convert DataFrame to list of dicts for Transformer
    train_data = list(train_data.to_dict('records'))
    
    data_train_all_nodes = train_data

    writer = SummaryWriter(os.path.join(file_path, 'lirpa_train_log'), flush_secs=10)

    print("Type of train_data:", type(train_data))
    if hasattr(train_data, 'shape'):  # DataFrame or tensor
        print("Shape of train_data:", train_data.shape)
        if hasattr(train_data, 'columns'):
            print("Columns:", list(train_data.columns))
            print("First row:\n", train_data.iloc[0])
        else:
            print("Dtype:", train_data.dtype)
    elif hasattr(train_data, '__len__'):
        print("Length of train_data:", len(train_data))
        if len(train_data) > 0:
            print("Type of first element (train_data[0]):", type(train_data[0]))
            if hasattr(train_data[0], 'shape'):  # Likely a tensor
                print("Shape of first element:", train_data[0].shape)
                print("Dtype of first element:", train_data[0].dtype)
            elif isinstance(train_data[0], (list, tuple)):
                print("Length of first element:", len(train_data[0]))
                print("First few elements of train_data[0]:", train_data[0][:5] if len(train_data[0]) > 5 else train_data[0])
            else:
                print("Value of first element:", train_data[0])
        else:
            print("train_data is empty")
    else:
        print("train_data:", train_data)

    

    model = Transformer(args_lirpa, train_data)

    #dev_batches = get_batches(dev_data, args_lirpa.batch_size)
    test_batches = get_batches(test_data, args_lirpa.batch_size)

    synonyms = generate_synonyms(idx_w)
    with open("data/synonyms.json", "w") as f:
        json.dump(synonyms, f, indent=4)
    ptb = PerturbationSynonym(budget=args_lirpa.budget)
    ptb.synonym = synonyms

    dummy_mask = torch.zeros(1, 1, 1, args_lirpa.max_sent_length, device=args_lirpa.device)
    dummy_embeddings = torch.zeros(1, args_lirpa.max_sent_length, args_lirpa.embedding_size, device=args_lirpa.device)
    dummy_embeddings = BoundedTensor(dummy_embeddings, ptb)
    model_ori = model.model_from_embeddings
    bound_opts = { 'activation_bound_option': args_lirpa.bound_opts_relu, 'exp': 'no-max-input', 'fixed_reducemax_index': True}
    if isinstance(model_ori, BoundedModule):
        model_bound = model_ori
    else:
        model_bound = BoundedModule(model_ori, (dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=args_lirpa.device)
    model.model_from_embeddings = model_bound
    if args_lirpa.loss_fusion:
        bound_opts['loss_fusion'] = True
        model_loss = BoundedModule(CrossEntropyWrapperMultiInput(model_ori), (torch.zeros(1, dtype=torch.long), dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=args_lirpa.device)
    else:
        model_loss = None

    ptb.model = model
    optimizer = model.build_optimizer()
    if args_lirpa.lr_decay < 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay)
    else:
        lr_scheduler = None
    if args_lirpa.robust:
        eps_scheduler = LinearScheduler(args_lirpa.eps, 'start={},length={}'.format(args_lirpa.eps_start, args_lirpa.eps_length))
        for i in range(model.checkpoint):
            eps_scheduler.step_epoch(verbose=False)
    else:
        eps_scheduler = None

    print("Model converted to bounded model")

    return (train_data, test_data, data_train_all_nodes, model, model_ori, model_loss, optimizer, lr_scheduler, eps_scheduler, dummy_embeddings, dummy_mask, bound_opts, ptb, test_batches, writer)

def generate_synonyms_complete(idx_w):
    return {x: (idx_w.tolist()) for x in idx_w}

def json_decoder(dict):
        return namedtuple('Args', dict.keys())(*dict.values())

def train_transformer_on_sorting(file_path):

    with open(Path(file_path) / "args_lirpa.json", "r") as f:
        args_lirpa = json.loads(f.read(), object_hook=json_decoder)

    (train_data, test_data, data_train_all_nodes, model, model_ori, model_loss, optimizer, lr_scheduler, eps_scheduler, dummy_embeddings, dummy_mask, bound_opts, ptb, test_batches, writer) = init_bounded_transformer(file_path, generate_synonyms_complete, args_lirpa)

    for t in range(model.checkpoint, args_lirpa.num_epochs):
        if t + 1 <= args_lirpa.num_epochs_all_nodes:
            d = train(args_lirpa, t + 1, get_batches(data_train_all_nodes, args_lirpa.batch_size), 'train', optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer)
        else:
            train(args_lirpa, t + 1, get_batches(train_data, args_lirpa.batch_size), 'train', optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer)
        #train(t + 1, dev_batches, 'dev', optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer)
        train(args_lirpa, t + 1, test_batches, 'test', optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer)

    torch.save(model.state_dict(), str(Path(file_path) / "model_lirpa_transformer.pt"))

def evaluate_sorting_lirpa_on_transformer(input_length, file_path):

    with open(Path(file_path) / "args_lirpa.json", "r") as f:
        args_lirpa = json.loads(str(f), object_hook=json_decoder)

    (train_data, test_data, data_train_all_nodes, model, model_ori, model_loss, optimizer, lr_scheduler, eps_scheduler, dummy_embeddings, dummy_mask, bound_opts, ptb, test_batches, writer) = init_bounded_transformer(file_path, generate_synonyms_complete, args_lirpa)
    model.load_state_dict(torch.load(Path(file_path) / "model_lirpa_transformer.pt", map_location=args_lirpa.device))

    for i in range(args_lirpa.num_epochs):
        eps_scheduler.step_epoch(verbose=False)
    res = []
    for i in range(1, args_lirpa.budget + 1):
        print('budget {}'.format(i))
        ptb.budget = i
        acc_rob = train(args_lirpa, None, test_batches, 'test', optimizer, eps_scheduler, lr_scheduler, model_loss, model, model_ori, dummy_embeddings, dummy_mask, bound_opts, ptb, writer)
        
        #for method in ['IBP', 'CROWN', 'alpha-CROWN']:
        method = "IBP+backward"
        start_time = time.time()
        lb, ub = model.compute_bounds(x=(dummy_embeddings,), method=method)
        end_time = time.time()
        print(f'{method} bounds: lower={lb.item()}, upper={ub.item()}, time={end_time - start_time:.4f}s')

        res.append(acc_rob)

    print('Verification results:')
    for i in range(len(res)):
        print('budget {}: robust accuracy {:.3f}'.format(i + 1, res[i]))
    print(res)

#------------LIRPA ORACLE---------------------
def evaluate_sorting_oracle_on_transformer(file_path):
    pass

def evaluate_sorting_oracle_on_transformer_program(file_path):
    pass

#--------------Z3---------------------

def build_perturbed_solver(solver_lib, input_length, enforce_start_end):
    N = input_length
    s = Solver()

    tokens = [Const(f"token_{i}", solver_lib.Token) for i in range(N)]
    tokens_ptb = [Const(f"token_ptb_{i}", solver_lib.Token) for i in range(N)]
    
    pos = [Int(f"pos_{i}") for i in range(N)]
    pos_ptb = [Int(f"pos_ptb_{i}") for i in range(N)] #needed?
    for i in range(N):
        s.add(pos[i] == IntVal(i))
        s.add(pos_ptb[i] == IntVal(i))

    outs, logits, predictions = solver_lib.build_pipeline(s, tokens, pos, N > 0 and enforce_start_end)
    outs_ptb, logits_ptb, predictions_ptb = solver_lib.build_pipeline(s, tokens_ptb, pos_ptb, N > 0 and enforce_start_end)

    return (s, tokens, tokens_ptb, pos, pos_ptb, outs, outs_ptb, logits, logits_ptb, predictions, predictions_ptb)

def check_Z3_robustness(budget, input_length, synonyms, solver, tokens, tokens_ptb, predictions, predictions_ptb):

    print(f"Z3 check budget {budget}")

    start_time = time.time()
    
    # perturbation constraints (Hamming distance <= budget)
    changes = [Bool(f'change_{i}') for i in range(input_length)]
    for i in range(input_length):
        for word, syns in synonyms:
            solver.add(Implies(tokens[i] == word, Or([tokens_ptb[i] == s for s in syns])))
        solver.add(changes[i] == (tokens_ptb[i] != tokens[i]))
    solver.add(Sum([If(changes[i], 1, 0) for i in range(input_length)]) <= budget)

    # output constraints (perturbed and original outputs differ)
    solver.add(Or([predictions_ptb[i] != predictions[i] for i in range(input_length)]))

    result = solver.check()
    end_time = time.time()
    print(f"Z3 check time: {end_time - start_time:.4f}s")
    if result == sat:
        m = solver.model()
        perturbed = [m.evaluate(tokens_ptb[i]) for i in range(input_length)]
        print(f"Not Robust")
        return False
    else:
        print("Robust")
        return True

def evaluate_sorting_Z3(input_length):

    #------Z3-------

    os.chdir(Path(__file__).parent.parent / "output" / "sort")
    solver_lib = importlib.import_module("output.sort.sort_Z3")

    print("Building Z3 solver...")
    start_time = time.time()
    solver, tokens, tokens_ptb, pos, pos_ptb, outs, outs_ptb, logits, logits_ptb, predictions, predictions_ptb = build_perturbed_solver(solver_lib, input_length, enforce_start_end=True)
    end_time = time.time()
    print(f"Z3 build time: {end_time - start_time:.4f}s")

    synonyms = {word : solver_lib.alphabet for word in solver_lib.alphabet}
    min, max = 0, input_length
    max_robust_budget = 0
    while min <= max:
        mid = (min + max) // 2
        robust = check_Z3_robustness(mid, input_length, synonyms, solver.__copy__(), tokens, tokens_ptb, predictions, predictions_ptb)
        if robust:
            max_robust_budget = mid
            min = mid + 1
        else:
            max_robust_budget = mid - 1
            max = mid - 1
        
    print(f"Maximum Z3 synonym replacement budget for robustness: {max_robust_budget}")

#-----------Main Program----------------

file_path = "output/sort"
if True:
    train_transformer_on_sorting(file_path)
if False:
    evaluate_sorting_Z3(input_length=6)
if False:
    evaluate_sorting_lirpa_on_transformer(input_length=6, file_path=file_path)