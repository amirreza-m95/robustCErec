import torch
import numpy as np
import random
import os
import argparse
import time
from methods.rcexplainer_helper import RuleMinerLargeCandiPool, evalSingleRule
from torch_geometric.utils import to_dense_adj
from methods.rcexplainer_helper import ExplainModule, train_explainer, evaluator_explainer
import data_utils
from tqdm import tqdm
import torch.nn.functional as F
from gnn_trainer import GNNTrainer

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse

# "convert a list of graph data and their corresponding node embeddings into a format suitable for training a model."
import torch
from torch_geometric.utils import to_dense_adj

def get_rce_format(data, node_embeddings):
    # Get the number of nodes for each graph
    num_nodes = [graph.num_nodes for graph in data]
    # Find the maximum number of nodes in the dataset
    max_num_nodes = max(num_nodes)
    # Extract labels for each graph
    label = [graph.y for graph in data]
    feat = []  # List to store padded node features
    adj = []   # List to store padded adjacency matrices
    node_embs_pads = []  # List to store padded node embeddings

    for i, graph in enumerate(data):
        # Pad node features to the size of the largest graph
        m = torch.nn.ZeroPad2d((0, 0, 0, max_num_nodes - graph.num_nodes))
        feat.append(m(graph.x))
        
        # Convert edge_index to dense adjacency matrix and pad it if necessary
        if graph.edge_index.shape[1] != 0:
            adj.append(to_dense_adj(graph.edge_index, max_num_nodes=max_num_nodes)[0])
        else:
            adj.append(torch.zeros(max_num_nodes, max_num_nodes))
        
        # Pad node embeddings to the size of the largest graph
        node_embs_pads.append(m(node_embeddings[i]))

    # Stack padded node features, adjacency matrices, and node embeddings
    adj = torch.stack(adj)
    feat = torch.stack(feat)
    label = torch.LongTensor(label)
    num_nodes = torch.LongTensor(num_nodes)
    node_embs_pads = torch.stack(node_embs_pads)

    # Return the formatted tensors
    return adj, feat, label, num_nodes, node_embs_pads

def extract_rules(model, train_data, preds, embs, device, pool_size=50):
    """
    Extracts rules from a given model using the provided training data, predictions, embeddings, and device.

    Args:
        model (torch.nn.Module): The model from which to extract rules.
        train_data (torch.Tensor): The training data used to train the model.
        preds (torch.Tensor): The predictions made by the model on the training data.
        embs (torch.Tensor): The embeddings of the training data.
        device (torch.device): The device on which the model and data are located.
        pool_size (int, optional): The size of the candidate pool. Defaults to 50.

    Returns:
        dict: A dictionary containing the extracted rules and their corresponding indices.

    """
    preds = np.ones(268)
    preds[0:100] = 0
    preds = torch.from_numpy(preds) # to be deleted later
    
    I = 36
    length = 2
    is_graph_classification = True
    pivots_list = []
    opposite_list = []
    rule_list_all = []
    cover_list_all = []
    check_repeat = np.repeat(False, len(train_data))
    rule_dict_list = []
    idx2rule = {}
    _pred_labels = torch.tensor(preds).cpu().numpy()
    num_classes = 2
    iter = 0
    for i in range(len(preds)):
        idx2rule[i] = None
    for i in range(len(preds)):
        rule_label = preds[i]

        if iter > num_classes * length - 1:
            if idx2rule[i] != None:
                continue

        if np.sum(check_repeat) >= len(train_data):
            break


        
       
        # Initialize the rule. Need to input the label for the rule and which layer we are using (I)
        rule_miner_train = RuleMinerLargeCandiPool(model, train_data, preds, embs, _pred_labels[i], device, I)

        feature = embs[i].float().unsqueeze(0).to(device)

        # Create candidate pool
        rule_miner_train.CandidatePoolLabel(feature, pool=pool_size)

        # Perform rule extraction
        array_labels = preds.long().cpu().numpy()  # train_data_upt._getArrayLabels()
        inv_classifier, pivots, opposite, initial = rule_miner_train.getInvariantClassifier(i, feature.cpu().numpy(), preds[i].cpu(), array_labels, delta_constr_=0)
        pivots_list.append(pivots)
        opposite_list.append(opposite)

        # saving info for gnnexplainer
        # rule_dict = {}
        # inv_bbs = inv_classifier._bb.bb

        # inv_invariant = inv_classifier._invariant
        # boundaries_info = []
        # b_count = 0
        # assert (len(opposite) == np.sum(inv_invariant))
        # for inv_ix in range(inv_invariant.shape[0]):
        #     if inv_invariant[inv_ix] == False:
        #         continue
        #     boundary_dict = {}
        #     # boundary_dict['basis'] = inv_bbs[:-1,inv_ix]
        #     boundary_dict['basis'] = inv_bbs[:, inv_ix]
        #     boundary_dict['label'] = opposite[b_count]
        #     b_count += 1
        #     boundaries_info.append(boundary_dict)
        # rule_dict['boundary'] = boundaries_info
        # rule_dict['label'] = rule_label.cpu().item()
        # print("Rules extracted: ", rule_dict)
        # rule_dict_list.append(rule_dict)
        # end saving info for gnn-explainer

        # evaluate classifier
        accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data, embs, preds)
        assert (cover_indi_train[i] == True)
        for c_ix in range(cover_indi_train.shape[0]):
            if cover_indi_train[c_ix] == True:
                if is_graph_classification:
                    idx2rule[c_ix] = len(rule_list_all)
                else:
                    if c_ix not in idx2rule:
                        idx2rule[c_ix] = []
                    idx2rule[c_ix].append(len(rule_list_all))

        rule_list_all.append(inv_classifier)
        cover_list_all.append(cover_indi_train)
        for j in range(len(train_data)):
            if cover_indi_train[j] == True:
                check_repeat[j] = True
        iter += 1

    rule_dict_save = {}
    rule_dict_save['rules'] = rule_dict_list
    rule_dict_save['idx2rule'] = idx2rule
    return rule_dict_save

def create_subgraphs_3_hops_away(dataset):
    import networkx as nx
    interaction_data = dataset.inter_feat
    user_ids = interaction_data['user_id']
    item_ids = interaction_data['item_id']
    edges = [(f'u{user_id}', f'i{item_id}') for user_id, item_id in zip(user_ids, item_ids)]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    from networkx.algorithms import bipartite
    assert bipartite.is_bipartite(graph), "The graph is not bipartite"

    # Extract 3-hop subgraphs for each node
    uniqueUserIds = [f'u{user_id}' for user_id in user_ids.unique()]

    subgraphs = []
    for node in uniqueUserIds:
        # Find all nodes 3 hops or fewer away from the current node
        nodes_within_3_hops = nx.single_source_shortest_path_length(graph, node, cutoff=2)
        
        # Extract the nodes (ignoring the distance, hence the .keys())
        uniqueUserIds = list(nodes_within_3_hops.keys())
        
        # Create a subgraph with these nodes
        subgraph = graph.subgraph(uniqueUserIds)
        
        subgraphs.append(subgraph)
        
    return subgraphs

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--opt_scheduler', default='none')
    parser.add_argument('--use_tuned_parameters', action='store_true')
    parser.add_argument('--lambda_', default=1.0, type=float, help="The hyperparameter of L_same, (1 - lambda) will be for L_opp")
    parser.add_argument('--mu_', default=0.0, type=float, help="The hyperparameter of L_entropy, makes the weights more close to 0 or 1")
    parser.add_argument('--beta_', default=0.0000001, type=float, help="The hyperparameter of L_sparsity, makes the explanations more sparse")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='lastfm-1k',
                        choices=['Mutagenicity', 'lastfm-1k', 'ml-1m'],
                        help="Dataset name")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gnn_run', type=int, default=1)
    parser.add_argument('--explainer_run', type=int, default=1)
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--epochs', type=int, default=5) # 20
    parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")
    parser.add_argument('--run', default="train", choices=['train', 'explain', 'recbole_hyper'])
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--model', default='LightGCN')
    parser.add_argument('--config_file_list', nargs='+', default=['config/lastfm-1k.yaml'])
    parser.add_argument('--config_dict', default=None)

    args = parser.parse_args()
    return args

def count_unique_and_sorted_occurrences(tuples_list):
    # Create a dictionary to store counts of unique second parts
    second_part_counts = {}
    
    # Iterate through each tuple in the list
    for tuple_item in tuples_list:
        # Extract the second part of the tuple
        second_part = tuple_item[1]
        
        # Increment the count for the second part in the dictionary
        if second_part in second_part_counts:
            second_part_counts[second_part] += 1
        else:
            second_part_counts[second_part] = 1
    
    # Sort the dictionary by values in descending order
    sorted_occurrences = dict(sorted(second_part_counts.items(), key=lambda item: item[1], reverse=True))
    
    # Return the count of unique second parts and the sorted dictionary of occurrences
    unique_count = len(sorted_occurrences)
    return unique_count, sorted_occurrences

def filter_tuples_by_id(tuples_list, id_value):
    filtered_tuples = [tuple_item for tuple_item in tuples_list if tuple_item[0] == id_value]
    return filtered_tuples

def find_least_connected_common_items(user_id1, user_id2, edges, unique_items, sortedValofConnections):
    """
    Function to find all the connected items of two given nodes of users and then 
    find the common items between them and check which of those common items have least connections with other users

    input varialbles description:
    user_id1: the first user id
    user_id2: the second user id
    edges: the list of edges between users and items
    unique_items: the unique items in the dataset
    sortedValofConnections: the sorted dictionary of occurrences of items
    """

    # Find the connected items of the two given users
    connected_items_user1 = [item_id for user_id, item_id in edges if user_id == user_id1]
    connected_items_user2 = [item_id for user_id, item_id in edges if user_id == user_id2]
    
    # Find the common items between the two users
    common_items = set(connected_items_user1).intersection(set(connected_items_user2))
    
    # Find the least connected item among the common items
    least_connected_item = min(common_items, key=lambda item: sortedValofConnections[item])
    
    return least_connected_item


import torch.nn as nn
class RecommendationModel(nn.Module):
    def __init__(self, num_nodes):
        super(RecommendationModel, self).__init__()
        self.fc = nn.Linear(num_nodes, num_nodes, bias=False)

    def forward(self, x):
        return self.fc(x)
    
# Mask learning via gradient ascent
def learn_mask(adj_matrix, epochs=50):
    mask = torch.rand(adj_matrix.size(), requires_grad=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        masked_adj = adj_matrix * mask
        outputs = model(masked_adj)
        loss = criterion(outputs, torch.eye(num_nodes))  # Example target: identity matrix for simplicity
        loss.backward()
        mask.data += 0.01 * mask.grad.data  # Gradient ascent on mask
        mask.grad.zero_()
    return mask.detach()

# Generate counterfactuals by altering important connections
def generate_counterfactual(user_index, mask, threshold=0.5):
    important_edges = (mask[user_index] > threshold).nonzero(as_tuple=True)[0]
    counterfactuals = []
    for edge in important_edges:
        original_value = adj_matrix[user_index, edge].clone()
        # Flip the connection
        adj_matrix[user_index, edge] = 1 - adj_matrix[user_index, edge]
        # Predict with altered connection
        altered_prediction = model(adj_matrix)
        adj_matrix[user_index, edge] = original_value  # restore original graph
        counterfactuals.append((edge, altered_prediction[user_index].detach()))
    return counterfactuals

class RecommenderSystem(torch.nn.Module):
    def __init__(self, config, train_data, valid_data, test_data):
        super(RecommenderSystem, self).__init__()
        self.config = config
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.device = config['device']
        self.model = get_model(config['model'])(config, train_data._dataset).to(self.device)
        self.trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, self.model)

    def train_model(self):
        # Set seeds for reproducibility
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])

        # Train the model
        best_valid_score, best_valid_result = self.trainer.fit(
            self.train_data,
            self.valid_data,
            saved=False,
            show_progress=self.config['show_progress'],
            verbose=True
        )

        # Switch model to evaluation mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def recommend(self, top_k=10):
        user_embeddings = self.trainer.model.user_embedding.weight.data
        item_embeddings = self.trainer.model.item_embedding.weight.data

        recommendations = {}
        for user_id in self.test_data:
            user_emb = user_embeddings[user_id[0]['user_id']]
    #       scores = torch.matmul(user_emb.unsqueeze(0), item_embeddings.t()).squeeze(0)
    #       top_items = torch.topk(scores, k=top_k).indices.tolist()
            similarity_scores = torch.cosine_similarity(item_embeddings, user_emb.unsqueeze(0), dim=-1)
            top_items = torch.topk(similarity_scores, k=top_k).indices.tolist()
            recommendations[user_id[0]['user_id']] = top_items
        
        return recommendations

def bipartite_graph(dataset):
    from networkx.algorithms import bipartite
    user_ids = dataset.inter_feat.iloc[:,0]
    item_ids = dataset.inter_feat.iloc[:,1]
    edges = [(f'u{user_id}', f'i{item_id}') for user_id, item_id in zip(user_ids, item_ids)]
    
    import networkx as nx
    BG = nx.Graph()
    nodes_set_user = [f'u{user_id}' for user_id in user_ids.unique()]
    nodes_set_item = [f'i{item_id}' for item_id in item_ids.unique()]

    BG.add_nodes_from(nodes_set_user, bipartite=0)
    BG.add_nodes_from(nodes_set_item, bipartite=1)

    edges = [(f'u{user_id}', f'i{item_id}') for user_id, item_id in zip(user_ids, item_ids)]
    BG.add_edges_from(edges)


    # Getting the bipartite nodes with bipartite attribute 0
    # bipartite_nodes_0 = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    from networkx.algorithms import bipartite
    assert bipartite.is_bipartite(BG), "The graph is not bipartite"
    weighted_projected_graph = bipartite.weighted_projected_graph(BG, nodes_set_user)

    return BG, weighted_projected_graph, edges


def load_dataset():
    # Load the dataset from RecBole (code from RS-BGRec repository)
    from recbole.config import Config
    config = Config(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list, config_dict=args.config_dict)
    config['data_path'] = os.path.join(config.file_config_dict['data_path'], config.dataset)
    from data.dataset import Dataset
    dataset = Dataset(config)
    # dataset.inter_feat => tensor([[  0,   0], [  0,   1], [  0,   2], ...]) user_id, item_id pairs [200586 rows x 2 columns]
    # dataset.inter_feat.iloc[:,0].nunique() # 269 unique user_ids # dataset.inter_feat.iloc[:,1].nunique() # 51610 unique item_ids
    # dataset.inter_feat.iloc[:,0].value_counts() # user_id counts
    return dataset, config

# function to remove a specific user,item pair from the dataset
def remove_user_item_pair(dataset, user_id, item_id):
    
    dataset.inter_feat = dataset.inter_feat[~((dataset.inter_feat['user_id'] == user_id) & (dataset.inter_feat['item_id'] == item_id))]
    dataset.inter_feat = dataset.inter_feat[~((dataset.inter_feat['user_id'] == item_id) & (dataset.inter_feat['item_id'] == user_id))]

    return dataset


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset, config = load_dataset()

    splits, indices = data_utils.split_data(dataset)
    train_indices = indices[0] # len 214
    val_indices = indices[1] # 26

    # 3hop subgraph extraction ---------------------------------------------------
    # subgraphs = create_subgraphs_3_hops_away(dataset)

    # # Create bipartite networkx graph --------------------------------------------
    # BG, weighted_projected_graph, edges  = bipartite_graph(dataset)

    # # nx.get_edge_attributes(weighted_projected_graph, 'weight') #('u1', 'u109'): 7
    # unique_items, sortedValofConnections = count_unique_and_sorted_occurrences(edges)
    # filterd_tuples = filter_tuples_by_id(edges, 'u1')
    # ----------------------------------------------------------------------------

    # from sklearn.model_selection import train_test_split
    # trainD, validtest = train_test_split(dataset.inter_feat, test_size=0.2, random_state=42)
    # validD, testD = train_test_split(validtest, test_size=0.5, random_state=42)
    # trainD.shape # (160468, 2) # validD.shape # (20059, 2) # testD.shape # (20059, 2)

    # dataset spliting
    from recbole.data import data_preparation
    dataset_copy = dataset.copy(new_inter_feat=dataset.inter_feat)
    train_data, valid_data, test_data = data_preparation(config, dataset_copy)


    from recbole.utils import get_trainer, get_model
    model = get_model(config['model'])(config, train_data._dataset).to(config['device'])
    # LightGCN(
    #   (user_embedding): Embedding(269, 64)
    #   (item_embedding): Embedding(51610, 64)
    #   (mf_loss): BPRLoss()
    #   (reg_loss): EmbLoss()
    # )

    # Trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    hyper = False # currently set to false
    
    best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data,
    saved=False, # currently set to true
    show_progress = config['show_progress'] and not hyper,
    verbose=not hyper
    )

    torch.manual_seed(args.explainer_run)
    torch.cuda.manual_seed(args.explainer_run)
    np.random.seed(args.explainer_run)
    random.seed(args.explainer_run)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # test_result = trainer.evaluate(test_data, load_best_model=True)
    # print("Test result: ", test_result)

    user_embeddings = trainer.model.user_embedding.weight.data # torch.Size([269, 64])
    item_embeddings = trainer.model.item_embedding.weight.data # torch.Size([51610, 64])
    # adj = trainer.model.norm_adj_matrix[51879][51879]

    preds = []
    preds30 = []
    for user_id in test_data:
        user_emb = user_embeddings[user_id[0]['user_id']]
        # Compute cosine similarity between user and all items
        # similarity_scores = torch.matmul(item_embeddings, user_emb.unsqueeze(1)).squeeze(1)
        similarity_scores = F.cosine_similarity(item_embeddings, user_emb.unsqueeze(0), dim=-1)
        top_items = torch.topk(similarity_scores, k=10).indices.tolist()[0]
        top30_items = torch.topk(similarity_scores, k=30).indices.tolist()[0]
        preds.append(top_items)
        preds30.append(top30_items)


    # Graph Embeddings
    concatenated_embeddings = torch.cat((user_embeddings, item_embeddings), dim=0)
    graph_embeddings=concatenated_embeddings

    # setting seed again because of rule extraction
    # torch.manual_seed(args.explainer_run)
    # torch.cuda.manual_seed(args.explainer_run)
    # np.random.seed(args.explainer_run)
    # random.seed(args.explainer_run)

    # adj, feat, label, num_nodes, node_embs_pads = get_rce_format(dataset, node_embeddings) 

    explainer = ExplainModule(
    num_nodes=269,
    # emb_dims=model.dim * 2,  # gnn_model.num_layers * 2,
    emb_dims=model.n_layers * 2,
    device=device,
    args=args
    )

    adj_matrix = nx.adjacency_matrix(weighted_projected_graph).todense() # torch.Size([269, 269])
    adj_matrix = torch.FloatTensor(adj_matrix)  #[  0.,   4.,   1.,  ...,  11.,  55.,   0.]
    

    num_nodes = len(weighted_projected_graph.nodes)
    # model = RecommendationModel(num_nodes)
    model = RecommenderSystem(config, train_data, valid_data, test_data)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    mask = learn_mask(adj_matrix, epochs=5)

    user_index = 0
    cf_examples = generate_counterfactual(user_index, mask, threshold=0.5)
    print(cf_examples)


    # rule_dict = {}
    # adj = nx.adjacency_matrix(G) #.todense()
    # feat = torch.zeros(269, 64)
    # label = torch.zeros(269)
    # num_nodes = torch.zeros(269)
    # node_embs_pads = torch.zeros(269, 269)


    # explainer, last_epoch = train_explainer(explainer, model, rule_dict, adj, feat, label, preds, 
    #                                         num_nodes, graph_embeddings, node_embs_pads, args,
    #                                         train_indices, val_indices, device)
    # all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(dataset)), device)
    # explanation_graphs = []
    # entered = 0
    # if (args.lambda_ != 0.0):
    #     counterfactual_graphs = []

    # for i, graph in enumerate(dataset):
    #     entered += 1
    #     explanation = all_explanations[i]
    #     explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
    #     edge_index = graph.edge_index
    #     edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]

    #     if (args.lambda_ != 0.0):
    #         d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
    #         c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
    #         explanation_graphs.append(d)
    #         counterfactual_graphs.append(c)
    #     else:
    #         # print('A')
    #         # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
    #         d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
    #         c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
    #         label = int(graph.y)
    #         pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
    #         pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
    #         explanation_graphs.append({
    #             "graph": d.cpu(), "graph_cf": c.cpu(),
    #             "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
    #         })


    print('finished')


def tempMain():

    
    explainer, last_epoch = train_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, args, train_indices, val_indices, device)
    all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(dataset)), device)
    explanation_graphs = []
    entered = 0
    if (args.lambda_ != 0.0):
        counterfactual_graphs = []
    for i, graph in enumerate(dataset):
        entered += 1
        explanation = all_explanations[i]
        explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
        edge_index = graph.edge_index
        edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]

        if (args.lambda_ != 0.0):
            d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
            c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
            explanation_graphs.append(d)
            counterfactual_graphs.append(c)
        else:
            # print('A')
            # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
            d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
            c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
            label = int(graph.y)
            pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
            pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
            explanation_graphs.append({
                "graph": d.cpu(), "graph_cf": c.cpu(),
                "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
            })

    print("check: ", entered, len(dataset))
    torch.save(explanation_graphs, explanations_path)
    if (args.lambda_ != 0.0):
        torch.save(counterfactual_graphs, counterfactuals_path)



def main():

    # Logging.-----------------------------------
    result_folder = f'data/{args.dataset}/rcexplainer_{args.lambda_}/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    from recbole.config import Config
    config = Config(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list, config_dict=args.config_dict)
    config['data_path'] = os.path.join(config.file_config_dict['data_path'], config.dataset)
    # print data path and config path
    print(config['data_path'])
    # start training by first loading the dataset
    from data.dataset import Dataset
    dataset = Dataset(config)

    # dataset spliting
    from recbole.data import data_preparation
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 3hop subgraph extraction
    import networkx as nx
    interaction_data = dataset.inter_feat
    user_ids = interaction_data['user_id'].numpy()
    item_ids = interaction_data['item_id'].numpy()
    # edges = list(zip(user_ids, item_ids))
    edges = [(f'u{user_id}', f'i{item_id}') for user_id, item_id in zip(user_ids, item_ids)]

    G = nx.Graph()
    G.add_edges_from(edges)
    from networkx.algorithms import bipartite
    assert bipartite.is_bipartite(G), "The graph is not bipartite"

    # Extract 3-hop subgraphs for each node
    subgraphs = create_subgraphs_3_hops_away(G)



    # Then model loading----------------------------
    from recbole.utils import init_seed, init_logger, set_color, get_trainer, get_model, get_local_time
    model = get_model(config['model'])(config, train_data._dataset).to(config['device'])
    # LightGCN(
    #   (user_embedding): Embedding(269, 64)
    #   (item_embedding): Embedding(51610, 64)
    #   (mf_loss): BPRLoss()
    #   (reg_loss): EmbLoss()
    # )

    # Trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
    trainer.saved_model_file = os.path.join(
        os.path.dirname(trainer.saved_model_file),
        '-'.join(split_saved_file[:1] + [dataset.dataset_name.upper()] + split_saved_file[1:])
            )

    hyper = False # currently set to false
    best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data,
    saved=True, # currently set to true
    show_progress = config['show_progress'] and not hyper,
    verbose=not hyper
    )

    # dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)

    torch.manual_seed(args.explainer_run)
    torch.cuda.manual_seed(args.explainer_run)
    np.random.seed(args.explainer_run)
    random.seed(args.explainer_run)

    best_explainer_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}.pt')
    args.best_explainer_model_path = best_explainer_model_path
    explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')
    counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}.pt')
    args.method = 'classification'

    # trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    # model = trainer.load(args.gnn_run)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # node_embeddings, graph_embeddings, outs = trainer.load_gnn_outputs(args.gnn_run)
    test_result = trainer.evaluate(test_data, load_best_model=True)
    print("Test result: ", test_result)

    user_embeddings = trainer.model.user_embedding.weight.data
    item_embeddings = trainer.model.item_embedding.weight.data
    preds = []
    preds30 = []
    for user_id in test_data:
        user_emb = user_embeddings[user_id[0]['user_id']]
        # Compute cosine similarity between user and all items
        # similarity_scores = torch.matmul(item_embeddings, user_emb.unsqueeze(1)).squeeze(1)
        similarity_scores = F.cosine_similarity(item_embeddings, user_emb.unsqueeze(0), dim=-1)
        top_items = torch.topk(similarity_scores, k=10).indices.tolist()[0]
        top30_items = torch.topk(similarity_scores, k=30).indices.tolist()[0]
        preds.append(top_items)
        preds30.append(top30_items)
        # preds = torch.argmax(outs, dim=-1)

    # add labels to subgraphs and modify them to contain node preds
    # for i, subgraph in enumerate(subgraphs):
        
    #     subgraph.graph['label'] = 1
    #     subgraph.nodes['label'] = torch.tensor(preds[i])
    #     subgraph.nodes['pred'] = torch.tensor(preds[i])
    #     subgraphs[i] = subgraph

    # train_indices = indices[0] 
    tensorofvalues = [i['user_id'] for i in train_data] # to be deleted later
    train_indices = [item for tensor in tensorofvalues for item in tensor.tolist()] # to be deleted later

    val_indices = indices[1]



    # rule extraction
    rule_folder = f'rcexplainer_rules/'
    if not os.path.exists(rule_folder):
        os.makedirs(rule_folder)
    rule_path = os.path.join(rule_folder, f'rcexplainer_{args.dataset}_{args.gnn_type}_rule_dict_run_{args.explainer_run}.npy')
    if os.path.exists(rule_path):
        rule_dict = np.load(rule_path, allow_pickle=True).item()
    else:
        concatenated_embeddings = torch.cat((user_embeddings, item_embeddings), dim=0)
        graph_embeddings=concatenated_embeddings
        # rule_dict = extract_rules(model, dataset, preds, graph_embeddings, device, pool_size=50) # min([100, (preds == 1).sum().item(), (preds == 0).sum().item()])
        # np.save(rule_path, rule_dict)

    # setting seed again because of rule extraction
    torch.manual_seed(args.explainer_run)
    torch.cuda.manual_seed(args.explainer_run)
    np.random.seed(args.explainer_run)
    random.seed(args.explainer_run)

    if args.dataset in ['Graph-SST2']:
        args.lr = args.lr * 0.05  # smaller lr for large dataset
        args.beta_ = args.beta_ * 10  # to make the explanations more sparse

    # adj, feat, label, num_nodes, node_embs_pads = get_rce_format(dataset, node_embeddings) 

    explainer = ExplainModule(
        num_nodes=269,
        # emb_dims=model.dim * 2,  # gnn_model.num_layers * 2,
        emb_dims=model.n_layers * 2,
        device=device,
        args=args
    )

    # to be removed later
    rule_dict = {}
    adj = torch.zeros(269, 269)
    feat = torch.zeros(269, 64)
    label = torch.zeros(269)
    num_nodes = torch.zeros(269)
    node_embs_pads = torch.zeros(269, 269)


    if args.dataset in ['Mutagenicity']:
        args.beta_ = args.beta_ * 30

    if args.dataset in ['NCI1']:
        args.beta_ = args.beta_ * 300

    if args.robustness == 'na':
        explainer, last_epoch = train_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, args, train_indices, val_indices, device)
        all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(dataset)), device)
        explanation_graphs = []
        entered = 0
        if (args.lambda_ != 0.0):
            counterfactual_graphs = []
        for i, graph in enumerate(dataset):
            entered += 1
            explanation = all_explanations[i]
            explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
            edge_index = graph.edge_index
            edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]

            if (args.lambda_ != 0.0):
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                explanation_graphs.append(d)
                counterfactual_graphs.append(c)
            else:
                # print('A')
                # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                label = int(graph.y)
                pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                explanation_graphs.append({
                    "graph": d.cpu(), "graph_cf": c.cpu(),
                    "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                })

        print("check: ", entered, len(dataset))
        torch.save(explanation_graphs, explanations_path)
        if (args.lambda_ != 0.0):
            torch.save(counterfactual_graphs, counterfactuals_path)
    elif args.robustness == 'topology_random':
        explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
        for noise in [1, 2, 3, 4, 5]:
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
            if (args.lambda_ != 0.0):
                counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
            explanation_graphs = []
            noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
            with torch.no_grad():
                node_embeddings = []
                graph_embeddings = []
                outs = []
                for graph in noisy_dataset:
                    node_embedding, graph_embedding, out = model(graph.to(device))
                    node_embeddings.append(node_embedding)
                    graph_embeddings.append(graph_embedding)
                    outs.append(out)
                graph_embeddings = torch.cat(graph_embeddings)
                outs = torch.cat(outs)
                preds = torch.argmax(outs, dim=-1)
            adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
            all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
            explanation_graphs = []
            if (args.lambda_ != 0.0):
                counterfactual_graphs = []
            for i, graph in enumerate(noisy_dataset):
                explanation = all_explanations[i]
                explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
                edge_index = graph.edge_index
                edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
                if (args.lambda_ != 0.0):
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                    explanation_graphs.append(d)
                    counterfactual_graphs.append(c)
                else:
                    # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    label = int(graph.y)
                    pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                    pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                    explanation_graphs.append({
                        "graph": d.cpu(), "graph_cf": c.cpu(),
                        "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                    })

            torch.save(explanation_graphs, explanations_path)
            if (args.lambda_ != 0.0):
                torch.save(counterfactual_graphs, counterfactuals_path)
    elif args.robustness == 'feature':
        explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
        for noise in [10, 20, 30, 40, 50]:
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
            if (args.lambda_ != 0.0):
                counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
            explanation_graphs = []
            noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_feature_dataset_name(dataset_name=args.dataset, noise=noise))
            with torch.no_grad():
                node_embeddings = []
                graph_embeddings = []
                outs = []
                for graph in noisy_dataset:
                    node_embedding, graph_embedding, out = model(graph.to(device))
                    node_embeddings.append(node_embedding)
                    graph_embeddings.append(graph_embedding)
                    outs.append(out)
                graph_embeddings = torch.cat(graph_embeddings)
                outs = torch.cat(outs)
                preds = torch.argmax(outs, dim=-1)
            adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
            all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
            explanation_graphs = []
            if (args.lambda_ != 0.0):
                counterfactual_graphs = []
            for i, graph in enumerate(noisy_dataset):
                explanation = all_explanations[i]
                explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
                edge_index = graph.edge_index
                edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
                if (args.lambda_ != 0.0):
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                    explanation_graphs.append(d)
                    counterfactual_graphs.append(c)
                else:
                    # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    label = int(graph.y)
                    pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                    pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                    explanation_graphs.append({
                        "graph": d.cpu(), "graph_cf": c.cpu(),
                        "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                    })

            torch.save(explanation_graphs, explanations_path)
            if (args.lambda_ != 0.0):
                torch.save(counterfactual_graphs, counterfactuals_path)
    elif args.robustness == 'topology_adversarial':
        explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
        for flip_count in [1, 2, 3, 4, 5]:
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
            if (args.lambda_ != 0.0):
                counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
            explanation_graphs = []
            noisy_dataset = data_utils.load_dataset(data_utils.get_topology_adversarial_attack_dataset_name(dataset_name=args.dataset, flip_count=flip_count))
            with torch.no_grad():
                node_embeddings = []
                graph_embeddings = []
                outs = []
                for graph in noisy_dataset:
                    node_embedding, graph_embedding, out = model(graph.to(device))
                    node_embeddings.append(node_embedding)
                    graph_embeddings.append(graph_embedding)
                    outs.append(out)
                graph_embeddings = torch.cat(graph_embeddings)
                outs = torch.cat(outs)
                preds = torch.argmax(outs, dim=-1)
            adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
            all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
            explanation_graphs = []
            if (args.lambda_ != 0.0):
                counterfactual_graphs = []
            for i, graph in enumerate(noisy_dataset):
                explanation = all_explanations[i]
                explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
                edge_index = graph.edge_index
                edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
                if (args.lambda_ != 0.0):
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                    explanation_graphs.append(d)
                    counterfactual_graphs.append(c)
                else:
                    # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                    d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                    label = int(graph.y)
                    pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                    pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                    explanation_graphs.append({
                        "graph": d.cpu(), "graph_cf": c.cpu(),
                        "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                    })

            torch.save(explanation_graphs, explanations_path)
            if (args.lambda_ != 0.0):
                torch.save(counterfactual_graphs, counterfactuals_path)
    else:
        raise NotImplementedError
