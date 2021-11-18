import argparse
parser = argparse.ArgumentParser("Energy Based Models")
# dataset / data root configuration
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
parser.add_argument("--data_root", type=str, default="../data")
parser.add_argument("--seed", type=int, default=658965, help="random seed")
parser.add_argument('--ratio_known_normal', default=0.05, type=float, help='Known ratio for normal data')
parser.add_argument('--ratio_known_outlier', default=0.05, type=float, help='Known ratio for outlier data')
parser.add_argument('--n_known_outlier', default=1, type=int, help='Number of known outlier classes')
parser.add_argument('--known_outlier', default=1, type=int, help='Known outlier class')
parser.add_argument('--known_normal', default=0, type=int, help='Known normal class')
parser.add_argument('--ratio_pollution', default=0, type=float, help='Ratio for polluted')

parser.add_argument("--aug", action="store_true", help="If true, use augmentation loss")
parser.add_argument("--ent", action="store_true", help="If true, use entropy loss")
parser.add_argument("--temperature", type=float, default=0.5)

# optimization
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                    help="decay learning rate by decay_rate at these epochs")
parser.add_argument("--decay_rate", type=float, default=.3,
                    help="learning rate decay multiplier")
parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--warmup_iters", type=int, default=-1,
                    help="number of iters to linearly increase learning rate, if -1 then no warmmup")

parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
parser.add_argument('--backbone', default='resnet', type=str, help='The backbone network')

# pretrained model loading
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument("--save_dir", type=str, default='./experiment') 
parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
parser.add_argument("--n_valid", type=int, default=5000)
## Randaugment parameters
parser.add_argument("--n",type=int,default=5)
parser.add_argument("--m",type=int,default=10)

##
parser.add_argument("--eval_no_norm", action="store_true", help="If true, do not use normalized feature to eval")
parser.add_argument("--set_initial_kmeanspp", default=True, help="If true, use kmeans++ to set initial prototypes")
parser.add_argument("--n_cluster",type=int, default=500, help="number of prototypes")
parser.add_argument("--sample_num",type=int, default=10, help="number of ensemble iteration")


args = parser.parse_args()

args.n_classes = 100 if args.dataset == "cifar100" else 10
