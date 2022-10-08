#!/usr/bin/env python
# -*- coding:utf8 -*-

import argparse
import torch
from sklearn import metrics
import numpy as np
from model import CacheGNN

random_seed = 1024
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


def predict(model, test_loader, train_loader, device, non_hop_2_adjacency_matrix_dict=None):
    model.eval()

    with torch.no_grad():
        training_embedding = []
        training_label = []

        for batch in train_loader:
            tag = batch.x.shape[0]
            batch = batch.to(device)
            mask = non_hop_2_adjacency_matrix_dict[tag]
            output, embedding = model.m_step_forward(batch,
                                                     mask)
            training_embedding.append(embedding)
            y = batch.y
            training_label.append(y)

        training_embedding = torch.concat(training_embedding)
        training_label = torch.cat(training_label)

    _train_lambda = model.lmabda_
    prob = _train_lambda / _train_lambda.sum()
    res = []
    sorted_prob, indices = torch.sort(prob, descending=True)
    sum_ = 0.

    for prob_i, id_i in zip(sorted_prob, indices):
        sum_ += prob_i.item()
        res.append(id_i.item())
        if sum_ >= 0.9:
            break

    _optimized_training_embedding = torch.index_select(training_embedding, 0,
                                                       torch.tensor(res).to(
                                                           _device))
    _optimized_training_label = torch.index_select(training_label, 0,
                                                   torch.tensor(res).to(
                                                       _device))

    ys, preds = [], []

    for test_data in test_loader:
        test_data = test_data.to(_device)
        y = test_data.y
        ys.append(y)
        with torch.no_grad():
            output = model.inference(test_data,
                                     _optimized_training_embedding,
                                     _optimized_training_label)
            preds.append((output > 0.5).float().cpu())

    gold, pred = torch.cat(ys, dim=0).cpu(), torch.cat(preds, dim=0).numpy()
    micro_f1 = metrics.f1_score(gold, pred, average='micro') * 100
    return micro_f1


def get_args():

    parser = argparse.ArgumentParse()
    parser.add_argument('--cuda', action='store_true',
                        default=True, help='Use CUDA or not')
    parser.add_argument('--cuda_id', type=int, required=True,
                        help='Which cuda index used')
    parser.add_argument('--hidden_dim',
                        type=int, required=True)
    parser.add_argument('--model',
                        type=str, required=True,
                        help='Model name')
    parser.add_argument('--dataset',
                        type=str, required=True,
                        help='Which dataset needed to predict')
    parser.add_argument('--train_model_path', type=str,
                        required=True, help='Trained CacheGNN model')
    parser.add_argument('--eta', type=float, required=True,
                        help='Trade-off parameters')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of global similar nodes')
    parser.add_argument('--criterion', type=str, required=True,
                        help='softmax or sigmoid')

    return parser.parser_args()


if __name__ == '__main__':

    args = get_args()

    device = torch.device(
        "cuda:{}".format(str(args.cuda_id)) if torch.cuda.is_available()
                                               and args.cuda else 'cpu')

    if args.dataset == 'ppi':
        train_dataset = PPI("./data/PPI", split='test')
        test_dataset = PPI("./data/PPI", split='test')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=2,
                                 shuffle=False)

        non_hop_2_adjacency_matrix_dict_path = os.path.join(
            "./data/PPI/non_hop_2_adjacency_matrix_dict_val_batch_size_2.pkl")
        train_node_id_dict_path = os.path.join("./data/PPI",
                                               "train_node_id_dict.pkl")


    model = CacheGNN(
        input_dim=train_dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes,
        nodes_numbers=num_nodes,
        normalize=True,
        k=args.k,
        eta=args.eta,
        device=device,
        model_name=args.model,
        criterion=args.criterion).to(device)

    predicted_micro_f1 = predict(model, test_loader, train_loader, device, non_hop_2_adjacency_matrix_dict)

