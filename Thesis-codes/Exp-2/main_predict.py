import argparse
import warnings
from main_load_predict import *
warnings.simplefilter("ignore", UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler_status", type=bool, default=True)
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--Exp_num", type=int, default=3)
    parser.add_argument("--num_dataset", type=int, default=10) # max = 10
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss_1",type=str,default="save_loss_GRU/")
    parser.add_argument("--path_to_save_loss_2",type=str,default="save_loss_RNN/")
    parser.add_argument("--path_to_save_loss_3",type=str,default="save_loss_LSTM/")
    parser.add_argument("--path_to_save_loss_4",type=str,default="save_loss_GRU_anchor/")
    parser.add_argument("--path_to_save_loss_5",type=str,default="save_loss_GRU_attn/")
    parser.add_argument("--path_to_save_loss_6",type=str,default="save_loss_RNN_attn/")
    parser.add_argument("--path_to_save_loss_7",type=str,default="save_loss_LSTM_attn/")
    parser.add_argument("--path_to_save_loss_8",type=str,default="save_loss_LSTM_anchor/")
    parser.add_argument("--path_to_save_loss_9",type=str,default="save_loss_RNN_anchor/")
    parser.add_argument("--path_to_save_loss_10",type=str,default="save_loss_Transformer/")
    parser.add_argument("--path_to_save_predictions_1",type=str,default="save_predictions_GRU/")
    parser.add_argument("--path_to_save_predictions_2",type=str,default="save_predictions_RNN/")
    parser.add_argument("--path_to_save_predictions_3",type=str,default="save_predictions_LSTM/")
    parser.add_argument("--path_to_save_predictions_4",type=str,default="save_predictions_GRU_anchor/")
    parser.add_argument("--path_to_save_predictions_5",type=str,default="save_predictions_GRU_attn/")
    parser.add_argument("--path_to_save_predictions_6",type=str,default="save_predictions_RNN_attn/")
    parser.add_argument("--path_to_save_predictions_7",type=str,default="save_predictions_LSTM_attn/")
    parser.add_argument("--path_to_save_predictions_8",type=str,default="save_predictions_LSTM_anchor/")
    parser.add_argument("--path_to_save_predictions_9",type=str,default="save_predictions_RNN_anchor/")
    parser.add_argument("--path_to_save_predictions_10",type=str,default="save_predictions_Transformer/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        lr=args.lr,
        frequency=args.frequency,
        Exp_num=args.Exp_num,
        num_dataset=args.num_dataset,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss_1=args.path_to_save_loss_1,
        path_to_save_loss_2=args.path_to_save_loss_2,
        path_to_save_loss_3=args.path_to_save_loss_3,
        path_to_save_loss_4=args.path_to_save_loss_4,
        path_to_save_loss_5=args.path_to_save_loss_5,
        path_to_save_loss_6=args.path_to_save_loss_6,
        path_to_save_loss_7=args.path_to_save_loss_7,
        path_to_save_loss_8=args.path_to_save_loss_8,
        path_to_save_loss_9=args.path_to_save_loss_9,
        path_to_save_loss_10=args.path_to_save_loss_10,
        path_to_save_predictions_1=args.path_to_save_predictions_1,
        path_to_save_predictions_2=args.path_to_save_predictions_2,
        path_to_save_predictions_3=args.path_to_save_predictions_3,
        path_to_save_predictions_4=args.path_to_save_predictions_4,
        path_to_save_predictions_5=args.path_to_save_predictions_5,
        path_to_save_predictions_6=args.path_to_save_predictions_6,
        path_to_save_predictions_7=args.path_to_save_predictions_7,
        path_to_save_predictions_8=args.path_to_save_predictions_8,
        path_to_save_predictions_9=args.path_to_save_predictions_9,
        path_to_save_predictions_10=args.path_to_save_predictions_10,
        scheduler_status=args.scheduler_status,
        patience=args.patience,
        factor=args.factor,
        device=args.device,
    )

