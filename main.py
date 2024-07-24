import os
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

torch.set_default_tensor_type("torch.DoubleTensor")
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

import utils
import model
import val


def train(cfg=None):
    if cfg["logging"]:
        wandb.init(project=cfg["project_name"], name=cfg["model_name"], tags=cfg["tags"], config=cfg)

    RawTrainData = utils.load_train(path=cfg["train_path"], cfg=cfg)
    TrainData = utils.format_train(RawTrainData, cfg=cfg)

    TestData = utils.load_test(path=cfg["test_path"], cfg=cfg)

    TrainTestData = utils.load_test(path=cfg["train_path"], cfg=cfg)

    ValData = utils.load_val(path=cfg["val_path"], cfg=cfg)

    TrainLoader = []
    AdaptLoader = []
    for i in range(cfg["nc"]):
        FullSet = model.Dataset(TrainData[i].X, TrainData[i].Y, TrainData[i].C)

        l = len(FullSet)
        if cfg["shuffle"]:
            TrainSet, AdaptSet = random_split(FullSet, [int(2 / 3 * l), l - int(2 / 3 * l)])
        else:
            TrainSet = model.Dataset(TrainData[i].X[:int(2 / 3 * l)], TrainData[i].Y[:int(2 / 3 * l)], TrainData[i].C)
            AdaptSet = model.Dataset(TrainData[i].X[int(2 / 3 * l):], TrainData[i].Y[int(2 / 3 * l):], TrainData[i].C)

        trainloader = torch.utils.data.DataLoader(TrainSet, batch_size=cfg["phi_shot"], shuffle=cfg["shuffle"],
                                                  num_workers=cfg["num_workers"])
        adaptloader = torch.utils.data.DataLoader(AdaptSet, batch_size=cfg["K_shot"], shuffle=cfg["shuffle"],
                                                  num_workers=cfg["num_workers"])
        TrainLoader.append(trainloader)
        AdaptLoader.append(adaptloader)

    # Store the model class definition in an external file so they can be referenced outside this script
    Phi = model.Phi(config=cfg)
    H = model.H(config=cfg)

    criterion_phi = nn.MSELoss()
    if cfg["classification_loss"] == "cross-entropy":
        criterion_h = nn.CrossEntropyLoss()
    optimizer_h = optim.Adam(H.parameters(), lr=cfg["lr"]["h"])
    optimizer_phi = optim.Adam(Phi.parameters(), lr=cfg["lr"]["phi"])

    metrics = {"train": {"loss": [], "loss_phi": [], "loss_h": []},
               "test": {"loss": []},
               "plots": {},
               "epoch": {"train":[], "test": []}}
    loss_test_best = 100.
    log_a = []

    for epoch in range(cfg["epochs"]+1):

        # Randomize the order in which we train over the subdatasets
        arr = np.arange(cfg["nc"])
        np.random.shuffle(arr)

        # Running loss over all subdatasets
        running_loss_phi = 0.
        running_loss_h = 0.

        for i in arr:
            with torch.no_grad():
                adaptloader = AdaptLoader[i]
                kshot_data = next(iter(adaptloader))
                trainloader = TrainLoader[i]
                phi_data = next(iter(trainloader))

            optimizer_phi.zero_grad()

            # Least-square to get $a$ from K-shot phi_data
            X = kshot_data["input"]  # K X dim_x
            Y = kshot_data["output"]  # K X dim_y
            phi = Phi(X)  # K X dim_a
            phi_T = phi.transpose(0, 1)  # dim_a X K
            A = torch.inverse(torch.mm(phi_T, phi))  # dim_a X dim_a
            a = torch.mm(torch.mm(A, phi_T), Y)  # dim_a X dim_y
            if torch.norm(a, "fro") > cfg["gamma"]:
                a = a / torch.norm(a, "fro") * cfg["gamma"]
            # print("label = %i" % i)
            # print("A matrix: ", a)

            # Training the Phi Network
            inputs = phi_data["input"]  # B X dim_x
            labels = phi_data["output"]  # B X dim_y
            c_labels = phi_data["c"].type(torch.long)

            log_a.append([i, a.detach().numpy().flatten()])

            outputs = torch.mm(Phi(inputs), a)
            loss_phi = criterion_phi(outputs, labels)
            temp = Phi(inputs)

            loss_h = criterion_h(H(temp), c_labels)

            loss = loss_phi + cfg["alpha"] * loss_h
            loss.backward()
            optimizer_phi.step()

            # Training the Discriminator Network H
            if np.random.rand() <= cfg["frequency_h"]:
                optimizer_h.zero_grad()
                temp = Phi(inputs)

                loss_h = criterion_h(H(temp), c_labels)

                loss_h.backward()
                optimizer_h.step()

                # Spectral normalization
                if cfg["SN"]:
                    for param in H.parameters():
                        M = param.detach().numpy()
                        if M.ndim > 1:
                            s = np.linalg.norm(M, 2)
                            param.data = param / s

            # # Spectral normalization
            # if cfg["SN"] > 0:
            #     for param in Phi.parameters():
            #         M = param.detach().numpy()
            #         if M.ndim > 1:
            #             s = np.linalg.norm(M, 2)
            #             if s > cfg["SN"]:
            #                 param.data = param / s * cfg["SN"]

            running_loss_phi += loss_phi.item()
            running_loss_h += loss_h.item()

        # Save statistics
        metrics["train"]["loss_phi"].append(running_loss_phi/cfg["nc"])
        metrics["train"]["loss_h"].append(running_loss_h/cfg["nc"])
        metrics["train"]["loss"].append(
            metrics["train"]["loss_phi"][-1] + cfg["alpha"] * metrics["train"]["loss_h"][-1])
        metrics["epoch"]["train"].append(epoch)

        if epoch % cfg["test_freq"] == 0:
            running_loss_test = 0.
            for j in range(len(TestData)):
                loss_test = val.val(x=TestData[j].X, y=TestData[j].Y, Phi=Phi, H=H, cfg=cfg)["adapt_loss"]
                running_loss_test += loss_test

                if f"loss_{j + 1}" not in metrics["test"].keys():
                    metrics["test"][f"loss_{j + 1}"] = []
                metrics["test"][f"loss_{j + 1}"].append(loss_test)

            metrics["test"]["loss"].append(running_loss_test / len(TestData))
            metrics["epoch"]["test"].append(epoch)

            if metrics["test"]["loss"][-1] < loss_test_best:
                loss_test_best = metrics["test"]["loss"][-1]
                Phi_best = Phi
                H_best = H

        if cfg["prog_freq"] and epoch % cfg["prog_freq"] == 0:
            print("[%4d/%d]  Training Loss: %.4f  |  Test Loss: %.4f" %
                  (epoch, cfg["epochs"], metrics["train"]["loss"][-1], metrics["test"]["loss"][-1]))

        if epoch != 0 and epoch % cfg["plot_freq"] == 0:
            if "T-SNE" not in metrics["plots"].keys():
                metrics["plots"]["T-SNE"] = None
            fig3 = val.plot_a(log_a=log_a, n=10, cfg=cfg)
            metrics["plots"]["T-SNE"] = fig3

            for j in range(len(TestData)):
                if f"Test {j + 1} Error" not in metrics["plots"].keys():
                    metrics["plots"][f"Test {j + 1} Error"] = None
                C = int(TestData[j].C)
                for idx, i in enumerate(arr):
                    if C == i:
                        a = np.asarray(log_a[-3+idx][1]).reshape(cfg["not"], 3)
                print(j+1)
                fig1 = val.plot_errors(t=TestData[j].meta["t"], x=TestData[j].X, y=TestData[j].Y, Phi=Phi,
                                      idx_val_start=2000, idx_val_end=2000, a=a, cfg=cfg, title="Test")
                metrics["plots"][f"Test {j + 1} Error"] = fig1
                # if f"Val {j + 1} Classification" not in metrics["plots"].keys():
                #     metrics["plots"][f"Val {j + 1} Classification"] = None
                # fig2 = val.plot_class(t=TestData[j].meta["t"], X=TestData[j].X, Y=TestData[j].Y,
                #                       Phi=Phi, H=H, idx_val_start=1000, idx_val_end=4000, cfg=cfg)
                # metrics["plots"][f"Val {j + 1} Classification"] = fig2
            trajs = []
            e = 0
            for k in range(len(TrainTestData)):
                S = (int(TrainTestData[k].C), TrainTestData[k].meta["condition"])
                if S not in trajs:
                    trajs.append(S)
                    if f"Train {e + 1} Error" not in metrics["plots"].keys():
                        metrics["plots"][f"Train {e + 1} Error"] = None
                    for idx, i in enumerate(arr):
                        if S[1] == i:
                            a = np.asarray(log_a[-3 + idx][1]).reshape(cfg["not"], 3)
                    print(k+1)
                    fig1 = val.plot_errors(t=TrainTestData[k].meta["t"], x=TrainTestData[k].X, y=TrainTestData[k].Y, Phi=Phi,
                                           idx_val_start=4000, idx_val_end=4000, a=a, cfg=cfg, title="Train")
                    metrics["plots"][f"Train {e + 1} Error"] = fig1
                    e+=1
            for p in range(len(ValData)):
                C = int(ValData[p].C)
                for idx, i in enumerate(arr):
                    if C == i:
                        a = np.asarray(log_a[-3 + idx][1]).reshape(cfg["not"], 3)
                print(p+1)
                fig1 = val.plot_errors(t=ValData[p].meta["t"], x=ValData[p].X, y=ValData[p].Y, Phi=Phi,
                                       idx_val_start=50000, idx_val_end=50000, a=a, cfg=cfg, title="Validation")
                metrics["plots"][f"Val {p + 1} Error"] = fig1

        if cfg["logging"]:
            log = {"epoch": epoch,
                   "train/loss": metrics["train"]["loss"][-1],
                   "train/loss_phi": metrics["train"]["loss_phi"][-1],
                   "train/loss_h": metrics["train"]["loss_h"][-1]}
            if epoch % cfg["test_freq"] == 0:
                log["test/loss"] = metrics["test"]["loss"][-1]
                for j in range(len(TestData)):
                    if f"test/loss_{j + 1}" not in log.keys():
                        log[f"test/loss_{j + 1}"] = None
                    log[f"test/loss_{j + 1}"] = metrics["test"][f"loss_{j + 1}"][-1]
            if epoch != 0 and epoch % cfg["plot_freq"] == 0:
                if "T-SNE Plot of Linear Weights" not in log.keys():
                    log[f"T-SNE Plot of Linear Weights"] = None
                log[f"T-SNE Plot of Linear Weights"] = metrics["plots"]["T-SNE"]
                for idx, i in enumerate(arr):
                    log[f"A{i+1}_{epoch}"] = wandb.Table(data=log_a[-3+idx][1].reshape(cfg["not"], 3).tolist())
                for key in metrics["plots"].keys():
                    log[key] = metrics["plots"][key]
            wandb.log(log)

            if epoch != 0 and epoch % cfg["plot_freq"] == 0:
                model.save2torch(Phi=Phi, H=H, name=str(epoch), cfg=cfg)
                model.save2txt(model=Phi, name=str(epoch), cfg=cfg)

    if cfg["logging"]:
        data = []
        for j in range(len(TestData)):
            errors = val.val(x=TestData[j].X, y=TestData[j].Y, Phi=Phi, H=H, cfg=cfg)
            data.append(list(errors.values()))
        table = wandb.Table(columns=["Maximum Loss", "Average Loss", "Adaptation Loss"], data=data)
        wandb.log({f"Validation Loss Metrics": table})

        fig = val.plot_dist(data=TrainData, cfg=cfg)
        wandb.log({"Training Dataset Distribution": fig})

        wandb.run.summary["val/loss_best"] = loss_test_best
        wandb.finish()

    if cfg["save"]:
        model.save2torch(Phi=Phi_best, H=H_best, name="best", cfg=cfg)
        model.save2txt(model=Phi_best, name="best", cfg=cfg)
        utils.save_config(path=os.path.join("models", cfg["save_path"], cfg["model_name"]), name="config", config=cfg)


if __name__ == "__main__":
    cfg = utils.load_config(name="base.yaml")

    cfg["drone"] = "bebop"
    cfg["version"] = "V5.0"

    cfg["dataset"] = {"name": "CyberZoo", "version": "V4.0"}
    cfg["tags"] = [cfg["version"]]

    cfg["X"] = ["v_body", "q", "rpm"]
    cfg["Y"] = "fr"

    cfg["train_path"] = os.path.join("datasets", cfg["dataset"]["name"], cfg["dataset"]["version"], "logs", "train")
    cfg["test_path"] = os.path.join("datasets", cfg["dataset"]["name"], cfg["dataset"]["version"], "logs", "test")
    cfg["val_path"] = os.path.join("datasets", cfg["dataset"]["name"], cfg["dataset"]["version"], "logs", "val")

    cfg['save_path'] = os.path.join(cfg['dataset']['name'], f"{cfg['drone']}_{'-'.join(i.split('_')[0] for i in cfg['X'])}_{cfg['Y']}", cfg['version'])

    cfg['project_name'] = f"{cfg['dataset']['name']}_{'-'.join(i.split('_')[0] for i in cfg['X'])}_{cfg['Y']}"
    cfg['model_name'] = f"{time.strftime('%m_%d_%H_%M')}"

    train(cfg=cfg)

    if os.path.isdir("/home/mauro/PycharmProjects/Neural-Fly/wandb"):
        shutil.rmtree("/home/mauro/PycharmProjects/Neural-Fly/wandb")