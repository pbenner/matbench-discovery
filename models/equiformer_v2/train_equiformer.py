# %%
        
from equitrain import get_args_parser
from equitrain import train

# %%

def main():

    r = 4.5

    args = get_args_parser().parse_args()
    args.train_file = f'data-r{r}/train.h5'
    args.valid_file = f'data-r{r}/valid.h5'
    args.statistics_file = f'data-r{r}/statistics.json'
    args.output_dir = 'result'

    train(args)

# %%
if __name__ == "__main__":
    main()
