import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from lombScargle import LombScargleBatchMask
from common import *
import pandas as pd
import os
import os.path as op

def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=20, foldername="", wandb=None, model_name='', eval_at=None, test_loader=None, nsample=100, scaler = 1, mean_scaler=0):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = op.join(foldername, "model.pth")
        best_model_path = op.join(foldername, "best_model.pth")

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        
        # Evaluate the model on the test set at the epochs specified in eval_at
        if eval_at is not None and epoch_no != config["epochs"] - 1:
            if test_loader is None:
                raise ValueError("test_loader is None")
            if epoch_no in eval_at:
                evaluate(model, test_loader, nsample=nsample, foldername=foldername, plot_params=None, wandb=wandb, model_name=model_name, epoch=epoch_no, scaler=scaler, mean_scaler=mean_scaler)

        model.train()
        losses_sum = {}
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                if epoch_no >= config["closs_start"]:
                    loss, losses = model(train_batch, closs_w=config["closs_w"])
                else:
                    loss, losses = model(train_batch)

                for key in losses:
                    if key not in losses_sum:
                        losses_sum[key] = 0
                    losses_sum[key] += losses[key].item()
                losses_avg = { key: losses_sum[key] / batch_no for key in losses_sum }
                loss.backward()
                optimizer.step()

                #it.set_postfix(ordered_dict=losses_avg | {"epoch":epoch_no}, refresh=False)
                it.set_postfix(ordered_dict=losses_avg.update({"epoch":epoch_no}), refresh=False)
                if batch_no >= config["itr_per_epoch"]:
                    break
            lr_scheduler.step()
        wandb.log(losses_avg)

        losses_sum = {}
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        
                        loss, losses = model(valid_batch, is_train=0)
                        for key in losses:
                            val_key = f"val_{key}"
                            if val_key not in losses_sum:
                                losses_sum[val_key] = 0
                            losses_sum[val_key] += losses[key].item()
                        losses_avg = { key: losses_sum[key] / batch_no for key in losses_sum }
                        it.set_postfix(ordered_dict=losses_avg | {"epoch": epoch_no}, refresh=False)

            if best_valid_loss > losses_avg["val_loss"]:
                best_valid_loss = losses_avg["val_loss"]
                torch.save(model.state_dict(), best_model_path)
                print(f"best loss is updated to {best_valid_loss} at {epoch_no}")

            wandb.log(losses_avg)

    if foldername != "":
        torch.save(model.state_dict(), output_path)
    if os.path.exists(best_model_path) and config["save_strategy"]=='best':
        print('Loading best model')
        model.load_state_dict(torch.load(best_model_path))

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", plot_params=None, wandb=None, model_name='', epoch=None):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        mse_ls = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        #We should save the spectrum
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            if epoch is None:
                fname_output_samples = foldername + "/output_samples_nsample" + str(nsample) + ".pk"
            else:
                fname_output_samples = foldername + f"/output_samples_nsample{nsample}_e{epoch}.pk"

            with open(
                fname_output_samples, "wb"
                # foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
    recon_median = reconstruct_data(all_generated_samples, all_target, all_evalpoint, all_observed_point)
    loss_masked, loss_full, loss_masked_mae = compute_spectral_loss(recon_median, all_target, all_evalpoint, all_observed_point, all_observed_time, path=foldername, plot=(plot_params is not None))
    loss_maked_fap, loss_full_fap, loss_masked_mae_fap = compute_spectral_loss(recon_median, all_target, all_evalpoint, all_observed_point, all_observed_time, path=foldername, fap=True, plot=(plot_params is not None))
    dict_res = {}
    dict_res["RMSE"] = np.sqrt(mse_total / evalpoints_total)
    dict_res["MAE"] = mae_total / evalpoints_total
    dict_res["CRPS"] = CRPS
    dict_res["CRPS_sum"] = CRPS_sum
    dict_res["MSE_LS_masked"] = loss_masked
    dict_res["MSE_LS_full"] = loss_full
    dict_res["MSE_LS"] = loss_maked_fap
    dict_res["MAE_LS"] = loss_masked_mae_fap
    dict_res["MSE_LS_full_fap"] = loss_full_fap
    dict_res["MAE_LS_full"] = loss_masked_mae
    pd.DataFrame(dict_res, index=[foldername]).to_csv(foldername + "/results.csv")
    short_dict = {key: (dict_res[key]) for key in ["MAE", "CRPS", "MSE_LS", "MAE_LS", "MAE_LS_full"]}
    if epoch is None:
        fname_results_short = foldername + "/results_short.csv"
    else:
        fname_results_short = foldername + f"/results_short_e{epoch}.csv"
    pd.DataFrame(short_dict, index=['res']).round(4).T.to_csv(fname_results_short)
    for key in short_dict:
        if epoch is not None:
            wandb.run.summary[key + f'_e{epoch}'] = short_dict[key]
        else:
            wandb.run.summary[key] = short_dict[key]
    # wandb.run.summary(short_dict)
    for i in range(10):
        if plot_params is not None:
            nrows = plot_params["nrows"]
            ncols = plot_params["ncols"]
            figsize = plot_params["figsize"]
            plot_samples(all_generated_samples, all_target, all_evalpoint, all_observed_point, idx_sample=i, save_path=f'{foldername}/testplot_{i}.png', model_name=model_name, nrows=nrows, ncols=ncols, figsize=figsize)
        else:
            plot_samples(all_generated_samples, all_target, all_evalpoint, all_observed_point, idx_sample=i, save_path=f'{foldername}/testplot_{i}.png', model_name=model_name)
