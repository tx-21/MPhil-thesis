import argparse
import warnings
from main_load import *
warnings.simplefilter("ignore", UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--Exp_num", type=int, default=40)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss_1",type=str,default="save_loss_1/")
    parser.add_argument("--path_to_save_loss_2",type=str,default="save_loss_2/")
    parser.add_argument("--path_to_save_loss_3",type=str,default="save_loss_3/")
    parser.add_argument("--path_to_save_loss_4",type=str,default="save_loss_4/")
    parser.add_argument("--path_to_save_loss_5",type=str,default="save_loss_5/")
    parser.add_argument("--path_to_save_predictions_1",type=str,default="save_predictions_1/")
    parser.add_argument("--path_to_save_predictions_2",type=str,default="save_predictions_2/")
    parser.add_argument("--path_to_save_predictions_3",type=str,default="save_predictions_3/")
    parser.add_argument("--path_to_save_predictions_4",type=str,default="save_predictions_4/")
    parser.add_argument("--path_to_save_predictions_5",type=str,default="save_predictions_5/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        lr=args.lr,
        frequency=args.frequency,
        Exp_num=args.Exp_num,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss_1=args.path_to_save_loss_1,
        path_to_save_loss_2=args.path_to_save_loss_2,
        path_to_save_loss_3=args.path_to_save_loss_3,
        path_to_save_loss_4=args.path_to_save_loss_4,
        path_to_save_loss_5=args.path_to_save_loss_5,
        path_to_save_predictions_1=args.path_to_save_predictions_1,
        path_to_save_predictions_2=args.path_to_save_predictions_2,
        path_to_save_predictions_3=args.path_to_save_predictions_3,
        path_to_save_predictions_4=args.path_to_save_predictions_4,
        path_to_save_predictions_5=args.path_to_save_predictions_5,
        device=args.device,
    )

