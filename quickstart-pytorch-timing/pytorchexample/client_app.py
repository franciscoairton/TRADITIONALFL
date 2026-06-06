"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.timing_utils import Timer, write_log

# Flower ClientApp
app = ClientApp()


def _get_config_value(config, key, default):
    try:
        return config[key]
    except Exception:
        return default


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    cfg = msg.content["config"]
    execution_id = int(_get_config_value(cfg, "execution_id", 1))
    server_round = int(_get_config_value(cfg, "server_round", 0))
    log_dir = str(context.run_config.get("log-dir", "logs"))

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    # Load the model and initialize it with the received weights
    with Timer() as timer:
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    write_log(
        execution_id=execution_id,
        server_round=server_round,
        node_id=partition_id,
        stage="cliente_inicializa_modelo_recebido",
        elapsed_sec=timer.elapsed,
        details=f"num_partitions={num_partitions}",
        log_dir=log_dir,
    )

    # Load the data
    with Timer() as timer:
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
    write_log(
        execution_id=execution_id,
        server_round=server_round,
        node_id=partition_id,
        stage="cliente_carrega_dados",
        elapsed_sec=timer.elapsed,
        details=f"batch_size={batch_size}; exemplos={len(trainloader.dataset)}",
        log_dir=log_dir,
    )

    # Call the training function
    with Timer() as timer:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
    write_log(
        execution_id=execution_id,
        server_round=server_round,
        node_id=partition_id,
        stage="treinamento_por_cliente",
        elapsed_sec=timer.elapsed,
        details=(
            f"local_epochs={context.run_config['local-epochs']}; "
            f"lr={msg.content['config']['lr']}; train_loss={train_loss}"
        ),
        log_dir=log_dir,
    )

    # Construct and return reply Message
    with Timer() as timer:
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply = Message(content=content, reply_to=msg)
    write_log(
        execution_id=execution_id,
        server_round=server_round,
        node_id=partition_id,
        stage="envio_cloud_por_cliente",
        elapsed_sec=timer.elapsed,
        details="prepara ArrayRecord(model.state_dict()) para retorno ao servidor",
        log_dir=log_dir,
    )
    return reply


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
