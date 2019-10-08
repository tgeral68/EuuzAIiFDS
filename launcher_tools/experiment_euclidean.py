import argparse
import tqdm

import torch
import pytorch_categorical
from torch import optim
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from embedding_tools.euclidean_embeddings_graph import EuclideanEmbedding as PEmbed
from em_tools.euclidean_em import EuclideanEM as PEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from launcher_tools import logger
from optim_tools import optimizer
import random 
import numpy as np

parser = argparse.ArgumentParser(description='Start an experiment')
parser.add_argument('--init-lr', dest="init_lr", type=float, default=-1.0,
                    help="Learning rate for the first embedding step")
parser.add_argument('--lr', dest="lr", type=float, default=5e-1,
                    help="learning rate for embedding")
parser.add_argument('--init-alpha', dest="init_alpha", type=float, default=-1.0,
                    help="alpha for the first embedding step")
parser.add_argument('--alpha', dest="alpha", type=float, default=1e-2,
                    help="alpha for embedding")
parser.add_argument('--init-beta', dest="init_beta", type=float, default=-1.0,
                    help="beta for the first embedding step")
parser.add_argument('--beta', dest="beta", type=float, default=1.,
                    help="beta for embedding")
parser.add_argument('--gamma', dest="gamma", type=float, default=1e-1,
                    help="gamma rate for embedding")
parser.add_argument('--n-gaussian', dest="n_gaussian", type=int, default=2,
                    help="number of gaussian for EM algorithm")
parser.add_argument('--dataset', dest="dataset", type=str, default="karate",
                    help="dataset to use for the experiments")
parser.add_argument('--walk-lenght', dest="walk_lenght", type=int, default=20,
                    help="size of random walk")
parser.add_argument('--cuda', dest="cuda", action="store_true", default=False,
                    help="using GPU for operation")
parser.add_argument('--epoch', dest="epoch", type=int, default=2,
                    help="number of loops alternating embedding/EM")
parser.add_argument('--epoch-embedding-init', dest="epoch_embedding_init", type=int, default=100,
                    help="maximum number of epoch for first embedding gradient descent")
parser.add_argument('--epoch-embedding', dest="epoch_embedding", type=int, default=10,
                    help="maximum number of epoch for embedding gradient descent")
parser.add_argument('--id', dest="id", type=str, default="0",
                    help="identifier of the experiment")
parser.add_argument('--save', dest="save", action="store_true", default=True,
                    help="saving results and parameters")
parser.add_argument('--precompute-rw', dest='precompute_rw', type=int, default=-1,
                    help="number of random path to precompute (for faster embedding learning) if negative \
                        the random walks is computed on flight")
parser.add_argument('--context-size', dest="context_size", type=int, default=5,
                    help="size of the context used on the random walk")
parser.add_argument("--negative-sampling", dest="negative_sampling", type=int, default=10,
                    help="number of negative samples for loss O2")
parser.add_argument("--embedding-optimizer", dest="embedding_optimizer", type=str, default="sgd", 
                    help="the type of optimizer used for learning poincaré embedding")
parser.add_argument("--em-iter", dest="em_iter", type=int, default=10,
                    help="Number of EM iterations")
parser.add_argument("--size", dest="size", type=int, default=3,
                    help="dimenssion of the ball")
parser.add_argument("--batch-size", dest="batch_size", type=int, default=512,
                    help="Batch number of elements")
parser.add_argument("--seed", dest="seed", type=int, default=42,
                    help="the seed used for sampling random numbers in the experiment")     
parser.add_argument('--force-rw', dest="force_rw", action="store_false", default=True,
                    help="if set will automatically compute a new random walk for the experiment")                
args = parser.parse_args()

# set the seed for random sampling
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(a=args.seed)


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }

optimizer_dict = {"sgd": optim.SGD}

# if(args.save):
#     print("The following options are use for the current experiment ", args)
#     os.makedirs("RESULTS/"+args.id+"/", exist_ok=True)
#     logger_object = logger.JSONLogger("RESULTS/"+args.id+"/log.json")
#     logger_object.append(vars(args))

# check if dataset exists

if(args.dataset not in dataset_dict):
    print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

if(args.embedding_optimizer not in optimizer_dict):
    print("Optimizer " + args.embedding_optimizer + " does not exist, please select one of the following : ")
    print(list(optimizer_dict.keys()))
    quit()

if(args.init_lr <= 0):
    args.init_lr = args.lr
if(args.init_alpha < 0):
    args.init_alpha = args.alpha
if(args.init_beta < 0):
    args.init_beta = args.beta
# set the seed for random sampling
alpha, beta = args.init_alpha, args.init_beta

print("Loading Corpus ")
D, X, Y = dataset_dict[args.dataset]()
print("Creating dataset")
# index of examples dataset
dataset_index = corpora_tools.from_indexable(torch.arange(0,len(D),1).unsqueeze(-1))
print("Dataset Size -> ", len(D))



D.set_path(False)

# negative sampling distribution
frequency = D.getFrequency()**(3/4)
frequency[:,1] /= frequency[:,1].sum()
frequency = pytorch_categorical.Categorical(frequency[:,1])
# random walk dataset
d_rw = D.light_copy()
rw_log = logger.JSONLogger("ressources/random_walk.conf", mod="continue")
if(args.force_rw):
    key = args.dataset+"_"+str(args.context_size)+"_"+str(args.walk_lenght)+"_"+str(args.seed) 
    if(key in rw_log):

        try:
            print('Loading random walks from files')
            d_rw = torch.load(rw_log[key]["file"])
            print('Loaded')
        except:
            os.makedirs("/local/gerald/KMEANS_RESULTS/", exist_ok=True)
            d_rw.set_walk(args.walk_lenght, 1.0)
            d_rw.set_path(True)
            d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)
            torch.save(d_rw, "/local/gerald/KMEANS_RESULTS/"+key+".t7")
            rw_log[key] = {"file":"/local/gerald/KMEANS_RESULTS/"+key+".t7", 
                        "context_size":args.context_size, "walk_lenght": args.walk_lenght,
                        "precompute_rw": args.precompute_rw}            
    else:
        os.makedirs("/local/gerald/KMEANS_RESULTS/", exist_ok=True)
        d_rw.set_walk(args.walk_lenght, 1.0)
        d_rw.set_path(True)
        d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)
        torch.save(d_rw, "/local/gerald/KMEANS_RESULTS/"+key+".t7")
        rw_log[key] = {"file":"/local/gerald/KMEANS_RESULTS/"+key+".t7", 
                       "context_size":args.context_size, "walk_lenght": args.walk_lenght,
                       "precompute_rw": args.precompute_rw}
else:
    d_rw.set_walk(args.walk_lenght, 1.0)
    d_rw.set_path(True)
    d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)   
if(args.save):
    os.makedirs("/local/gerald/AISTAT_RESULTS/"+args.id+"/", exist_ok=True)
    logger_object = logger.JSONLogger("/local/gerald/AISTAT_RESULTS/"+args.id+"/log.json")
    logger_object.append(vars(args))
# neigbhor dataset
d_v = D.light_copy()
d_v.set_walk(1, 1.0)

print(d_rw[1][0].size())

print("Merging dataset")
embedding_dataset = corpora_tools.zip_datasets(dataset_index,
                                                corpora_tools.select_from_index(d_v, element_index=0),
                                                d_rw
                                                )
print(embedding_dataset[29][-1][20:25])
training_dataloader = DataLoader(embedding_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=8,
                            collate_fn=data_tools.PadCollate(dim=0),
                            drop_last=False
                    )

representation_d = []
pi_d = []
mu_d = []
sigma_d = []
# if dimension is 2 we can plot 
# we store colors here
if(args.size == 2):
    import numpy as np
    unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
    colors = []
    import matplotlib.pyplot as plt
    import matplotlib.colors as plt_colors
    import numpy as np
    for i in range(len(D.Y)):
        colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))


alpha, beta = args.init_alpha, args.init_beta
embedding_alg = PEmbed(len(embedding_dataset), size=args.size, lr=args.init_lr, cuda=args.cuda, negative_distribution=frequency,
                        optimizer_method=optimizer_dict[args.embedding_optimizer])
em_alg = PEM(args.size, args.n_gaussian, init_mod="kmeans-hyperbolic", verbose=True)
pi, mu, sigma = None, None, None
pik = None
epoch_embedding = args.epoch_embedding_init
for i in tqdm.trange(args.epoch):
    if(i==1):
        embedding_alg.set_lr(args.lr)
        alpha, beta = args.alpha, args.beta
        epoch_embedding = args.epoch_embedding

    embedding_alg.fit(training_dataloader, alpha=alpha, beta=beta, gamma=args.gamma, max_iter=epoch_embedding,
                        pi=pik, mu=mu, sigma=sigma, negative_sampling=args.negative_sampling)

representation_d.append(embedding_alg.get_PoincareEmbeddings().cpu())

#evaluate performances on all disc
#evaluate performances on all disc
total_accuracy = evaluation.accuracy_euclidean_kmeans(representation_d[0], D.Y,torch.zeros(args.n_gaussian), verbose=False)
print("\nPerformances  kmeans-> " ,
    total_accuracy
)
#evaluate performances on all disc
# total_accuracy = evaluation.accuracy_disc_kmeans(representation_d[0], D.Y, mu_d[0], verbose=False)
# print("\nPerformances  kmeans-> " ,
#     total_accuracy
# )
logger_object.append({"accuracy_kmeans": total_accuracy})

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
torch.save(representation_d, "/local/gerald/AISTAT_RESULTS/"+args.id+"/embeddings.t7")
# torch.save( {"pi": pi_d, "mu":mu_d, "sigma":sigma_d}, "RESULTS/"+args.id+"/pi_mu_sigma.t7")


# if(args.size == 2):

#     fig = plot_tools.euclidean_plot(representation_d[0], pi_d[0], mu_d[0],  sigma_d[0], 
#                                                 labels=None, grid_size=100, colors=colors, 
#                                                 path="RESULTS/"+args.id+"/fig.pdf")
#     # plt.savefig("RESULTS/"+args.id+"/fig.pdf", format="pdf")

# print({"pi": pi_d, "mu":mu_d, "sigma":sigma_d})
