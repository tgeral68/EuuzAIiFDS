import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from embedding_tools.poincare_embeddings_graph_multi import RiemannianEmbedding as PEmbed
from em_tools.poincare_em_multi import RiemannianEM as PEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from launcher_tools import logger
from optim_tools import optimizer

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="RESULTS/football-5D-KMEANS-1/",
                    help="embeddings location file")
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")              
args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
log_in = logger.JSONLogger(os.path.join(args.file,"log.json"), mod="continue")
dataset_name = log_in["dataset"]
print(dataset_name)
n_gaussian = log_in["n_gaussian"]
if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


print("Loading Corpus ")
D, X, Y = dataset_dict[dataset_name]()

results = []
std_kmeans = []
representations = torch.load(os.path.join(args.file,"embeddings.t7"))[0]

for i in tqdm.trange(args.n):
    total_accuracy, stdmx, stdmn, std = evaluation.accuracy_disc_kmeans(representations, D.Y, torch.zeros(n_gaussian),  verbose=False)
    results.append(total_accuracy)
    std_kmeans.append(std.tolist())

R = torch.Tensor(results)
S = torch.Tensor(std_kmeans)
log_in.append({"kmeans_from_embeddings": {"RES":R.tolist(), "MIN":R.min().item(),
                "MAX":R.max().item(), "MEANS":R.mean().item(), "STD": R.std().item(),
                "STD_KMEANS":S.tolist()}})
print("MEANS -> ", R.mean())
print("MAX -> ", R.max())
print("MIN -> ", R.min())
