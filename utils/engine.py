from timeit import default_timer as timer
import os
import time
import shutil
import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch as mp


def train_one_epoch(model, optimizer, data_loader):
    model.train()
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []

    for image, labels in iter(data_loader):
        optimizer.zero_grad()
        image, labels = image.to(device), labels.to(device)
        class_logits = model.forward(image)
        loss = F.cross_entropy(class_logits, labels)
        loss.backward()
        loss_list += [loss.tolist()]
        pred_lab = torch.argmax(class_logits, 1)
        label_match_list += (pred_lab == labels).tolist()
        optimizer.step()

    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(
        label_match_list, dtype=torch.float32)))

    return mean_loss, mean_accuracy


@torch.no_grad()
def eval_one_epoch(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []

    for image, labels in iter(data_loader):
        image, labels = image.to(device), labels.to(device)
        class_logits = model.forward(image)
        loss = F.cross_entropy(class_logits, labels)
        loss_list += [float(loss)]
        pred_lab = torch.argmax(class_logits, 1)
        label_match_list += (pred_lab == labels)

    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(
        label_match_list, dtype=torch.float32)))
    return mean_loss, mean_accuracy


def train_model(model,
                train_loader,
                test_loader,
                model_dir,
                verbose=True,
                params=None):

    model = model.to(params['device'])
    optimizer = torch.optim.SGD(
        model.parameters(), momentum=params['momentum'], lr=params['learning_rate'])

    try:
        experiment_id = mlflow.create_experiment(params['experiment_name'])
    except:
        current_experiment = dict(
            mlflow.get_experiment_by_name(params['experiment_name']))
        experiment_id = current_experiment['experiment_id']

    with mlflow.start_run(experiment_id=experiment_id):
        start_time = timer()
        t0 = time.time()
        best = 0
        for epoch in range(1, params['epochs'] + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, optimizer, train_loader)
            test_loss, test_accuracy = eval_one_epoch(
                model, test_loader)
                 
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy
                },
                step=epoch
            )

            if verbose:
                print(
                    f"Epoch {epoch}/{params['epochs']}\n loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {test_loss} - val_accuracy: {test_accuracy:.4f} - time: {time.time() - t0:.0f}s")

            if epoch == 1:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)

                with open(f"{model_dir}/train_log.csv", 'w', newline='\n', encoding='utf-8') as f:
                    f.write("train_loss,train_accuracy,valid_loss,valid_accuracy\n")

            with open(f"{model_dir}/train_log.csv", 'a', newline='\n', encoding='utf-8') as f:
                f.write(
                    f'{train_loss:.4f}, {train_accuracy:.4f}, {test_loss:.4f}, {test_accuracy:.4f}\n')
            torch.save(model.state_dict(),f"{model_dir}/weights_epoch_{epoch}.pt")
        end_time = timer()
        mp.log_model(model, "Model")
        mlflow.log_metrics({
            "time": end_time - start_time
        })
        mlflow.log_params(params)
        del model
        mlflow.end_run()