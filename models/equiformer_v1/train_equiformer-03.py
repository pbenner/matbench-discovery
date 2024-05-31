# %%
        
from equitrain import get_args_parser
from equitrain import train

# %%

def main():

    r = 5.0

    args = get_args_parser().parse_args()

    args.load_checkpoint_model = 'result-02/best_val_epochs@1_e@19616.7084/pytorch_model.bin'

    args.train_file       = f'data-r{r}/train.h5'
    args.valid_file       = f'data-r{r}/valid.h5'
    args.statistics_file  = f'data-r{r}/statistics.json'
    args.output_dir       = 'result-03'
    args.r_max            = r

    args.epochs           = 10
    args.batch_size       = 16
    args.batch_edge_limit = 100000
    args.lr               = 2e-5

    args.alpha_drop       = 0.2
    args.proj_drop        = 0.0
    args.drop_path_rate   = 0.0
    args.out_drop         = 0.0

    args.energy_weight    =   1.0
    args.force_weight     = 100.0
    args.stress_weight    = 200.0

    train(args)

# %%
if __name__ == "__main__":
    main()
