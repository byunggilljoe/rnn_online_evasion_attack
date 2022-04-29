import data, models
import argparse
from mnist_attack_experiment import MNISTAttackExperiment
from fashion_mnist_attack_experiment import FashionMNISTAttackExperiment
from udacity_attack_experiment import UdacityAttackExperiment
from sentiment_attack_experiment import SentimentAttackExperiment
from mortality_attack_experiment import MortalityAttackExperiment
from energy_attack_experiment import EnergyAttackExperiment
from user_attack_experiment import UserAttackExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--eps', type=float, default=None)
parser.add_argument('--step_size', type=float, default=None)
parser.add_argument('--iters', type=int, default=None)
parser.add_argument('--attack', type=str, choices=['greedy', 'predictive', 'clairvoyant', 'uniform_predictive',
                                                   'uniform_ocp_predictive', 'ocp_predictive', 'ocp_clairvoyant',
                                                   'IID_predictive', 'permuted_predictive', 'clairvoyant_k'])
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--t_start', type=int, default=0)
parser.add_argument('--t_end', type=int, default=28)

parser.add_argument('--t_eval_start', type=int, default=21)
parser.add_argument('--t_eval_end', type=int, default=28)

parser.add_argument('--eval_last', dest="eval_last", action="store_true")
parser.add_argument('--no-eval_last', dest="eval_last", action="store_false")
parser.set_defaults(eval_last=True)
parser.add_argument('--NUM_TEST_BATCH', type=int, default=5)

parser.add_argument('--max_attack', dest='max_attack', action='store_true')
parser.add_argument('--no-max_attack', dest='max_attack', action='store_false')
parser.set_defaults(max_attack=False)

parser.add_argument('--ts_ranged', dest='ts_ranged', action='store_true')
parser.add_argument('--no-ts_ranged', dest='ts_ranged', action='store_false')
parser.set_defaults(ts_ranged=False)
parser.add_argument('--ts_neg_gamma', type=float, default=0)

parser.add_argument('--transfer_victim_key', type=str, default=None)

parser.add_argument('--save_result_image', dest='save_result_image', action='store_true')
parser.set_defaults(save_result_image=False)

parser.add_argument('--num_sample', type=int, default=1)
parser.add_argument('--targeted', dest='targeted', action='store_true')
parser.set_defaults(targeted=False)

#for udacity
parser.add_argument("--root_dir", type=str, default="../udacity-data")
parser.add_argument("--batch_size", type=int, required=False, default=None)
parser.add_argument("--NUM_HISTORY", type=int, default=5)
parser.add_argument("--NUM_TOTAL", type=int, default=20)
parser.add_argument("--DOWNSAMPLE_FACTOR", type=int, default=2)

parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=4, type=int)
parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers') #################################
parser.add_argument('--g_dim', type=int, default=1024,
                help='dimensionality of encoder hput vector and decoder input vector') #################################

parser.add_argument("--NUM_BATCH", type=int, default=1)

parser.add_argument("--trial", type=int, default=0, required=True)

parser.add_argument('--transfer_attack', dest='transfer_attack', action='store_true')
parser.set_defaults(transfer_attack=False)

parser.add_argument('--noise_alpha', type=float, default=0)

parser.add_argument('--save_realtime_training_data', dest='save_realtime_training_data', action='store_true')
parser.set_defaults(save_realtime_training_data=False)

parser.add_argument("--NUM_WAVE", type=int, default=2, required=False)

args = parser.parse_args()




default_batch_size_dict={"mnist":128,
                   "fashion_mnist":128,
                   "mortality":64,
                   "udacity":32,
                   "sentiment":100,
                   "energy":128,
                   "user": 128}
                   
if args.batch_size == None:
    args.batch_size = default_batch_size_dict[args.dataset]

experiment_dict = {"mnist":MNISTAttackExperiment,
                   "fashion_mnist":FashionMNISTAttackExperiment,
                   "mortality":MortalityAttackExperiment,
                   "udacity":UdacityAttackExperiment,
                   "sentiment":SentimentAttackExperiment,
                   "energy":EnergyAttackExperiment,
                   "user": UserAttackExperiment}

experiment = experiment_dict[args.dataset](args)
experiment.load_data_and_model()
experiment.execute()
