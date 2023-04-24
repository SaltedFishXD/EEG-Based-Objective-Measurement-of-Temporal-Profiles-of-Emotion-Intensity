from training import *
from preprocess import *
from generate_profile import *
from score import *
from time_interval import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='/mnt/left/annie503/data_preprocessed_python/')
    parser.add_argument('--start-subjects', type=int, default=0)
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V'])
    #parser.add_argument('--sampling-rate', type=int, default=128)
    #parser.add_argument('--input-shape-baseline', type=tuple, default=(1, 32, 7680))
    parser.add_argument('--data-format', type=str, default='raw')
    ######## Training Process ########
    parser.add_argument('--round', type=int, default=3)
    parser.add_argument('--random-seed', type=int, default=44)
    parser.add_argument('--weight_decays', type=float, default=1e-2)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--testlist', type=bool, default=False)
    parser.add_argument('--testlist_file', default='./save/arousal/')
    parser.add_argument('--max-epoch', type=int, default=150)
    parser.add_argument('--ft-epoch', type=int, default=50)
    parser.add_argument('--threshold', type=int, default=20) 
    parser.add_argument('--retrain-type', type=str, default='ft', choices=['ft', 'nm'])
    parser.add_argument('--selection', type=str, default='time', choices=['score', 'time'])

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--gpu', default='0')
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='SCCNet')
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)

    ######## Reproduce the result using the saved model ######
    args = parser.parse_args()

    pd = PrepareData(args)
    pd.run(args.subjects)
    train_model = training(args)
    train_model.Run_CrossTrial(args.start_subjects, args.subjects, args.round, args.max_epoch, args.label_type, 
    args.weight_decays, args.save_path, args.testlist, args.testlist_file)
    profile = Profile(args)
    profile.generate(args.start_subjects, args.subjects)
    if args.selection == 'score':
        score = Score(args)
        score.Run(int(100-args.threshold))
    if args.selection == 'time':
        TIME = Time_Interval(args)
        TIME.Run(args.threshold)


