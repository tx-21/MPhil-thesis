import argparse
import warnings

from sklearn.inspection import partial_dependence
from main_load import *
warnings.simplefilter("ignore", UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler_status", type=bool, default=True)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=100) #100
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--Exp_num", type=int, default=3) #3
    parser.add_argument("--model_number", type=int, default=1) # max = 5
    parser.add_argument("--num_dataset", type=int, default=7) # max = 7
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss_1",type=str,default="save_loss_LSTM/")
    parser.add_argument("--path_to_save_predictions_1",type=str,default="save_predictions_LSTM/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        lr=args.lr,
        frequency=args.frequency,
        Exp_num=args.Exp_num,
        model_number=args.model_number,
        num_dataset=args.num_dataset,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss_1=args.path_to_save_loss_1,
        path_to_save_predictions_1=args.path_to_save_predictions_1,
        scheduler_status=args.scheduler_status,
        patience=args.patience,
        factor=args.factor,
        device=args.device,
    )

