"""pytorchexample: A Flower / PyTorch app with multiple timing executions."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.timing_utils import Timer, clear_execution_logs, compile_summary, get_log_dir, write_log

# Create ServerApp
app = ServerApp()


class TimedFedAvg(FedAvg):
    """FedAvg with timing logs for selection, send/receive, and aggregation."""

    def __init__(self, *, execution_id: int, log_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.execution_id = execution_id
        self.log_dir = log_dir

    def configure_train(self, server_round, arrays, config, grid):  # type: ignore[override]
        with Timer() as timer:
            messages = list(super().configure_train(server_round, arrays, config, grid))
            # Make the current round visible to each client log.
            for msg in messages:
                try:
                    msg.content["config"]["server_round"] = server_round
                    msg.content["config"]["execution_id"] = self.execution_id
                except Exception:
                    pass
        write_log(
            execution_id=self.execution_id,
            server_round=server_round,
            node_id="server",
            stage="seleciona_clientes",
            elapsed_sec=timer.elapsed,
            details=f"num_clientes_selecionados={len(messages)}",
            log_dir=self.log_dir,
        )
        return messages

    def aggregate_train(self, server_round, replies):  # type: ignore[override]
        replies_list = list(replies)
        with Timer() as timer:
            result = super().aggregate_train(server_round, replies_list)
        write_log(
            execution_id=self.execution_id,
            server_round=server_round,
            node_id="server",
            stage="agregacao_de_todos_clientes",
            elapsed_sec=timer.elapsed,
            details=f"num_respostas_agregadas={len(replies_list)}",
            log_dir=self.log_dir,
        )
        return result

    def start(self, *args, **kwargs):  # type: ignore[override]
        """Wrap start to log the complete execution time.

        This method also estimates the send-to-clients stage by measuring the
        complete strategy.start wall-clock time per execution. The per-round
        send time is logged inside _train_round when available in the Flower
        version. If the internal method changes, the total execution time is
        still recorded.
        """
        with Timer() as timer:
            result = super().start(*args, **kwargs)
        write_log(
            execution_id=self.execution_id,
            server_round=0,
            node_id="server",
            stage="tempo_total_execucao",
            elapsed_sec=timer.elapsed,
            details="tempo wall-clock total da execucao strategy.start",
            log_dir=self.log_dir,
        )
        return result

    # Flower 1.29 calls train rounds internally via this method. Keeping this
    # override isolated allows us to log the send/receive time without changing
    # the original aggregation logic. If a future Flower version changes this
    # internal method, the app will still run because this method is only used
    # when present in the parent class.
    def _train_round(self, server_round, arrays, train_config, grid):  # type: ignore[override]
        start = time.perf_counter()
        result = super()._train_round(server_round, arrays, train_config, grid)
        elapsed = time.perf_counter() - start
        num_responses = "unknown"
        try:
            replies = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            num_responses = len(replies) if replies is not None else "unknown"
        except Exception:
            pass
        write_log(
            execution_id=self.execution_id,
            server_round=server_round,
            node_id="server",
            stage="envia_para_clientes",
            elapsed_sec=elapsed,
            details=f"num_respostas={num_responses}; inclui_envio_treino_retorno=true",
            log_dir=self.log_dir,
        )
        return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    num_executions: int = int(context.run_config.get("num-executions", 1))
    warmup_rounds: int = int(context.run_config.get("warmup-rounds", 1))
    num_supernodes: int = int(context.run_config.get("num-supernodes", 10))
    num_selected_clients: int = int(context.run_config.get("num-selected-clients", 10))
    num_evaluate_clients: int = int(context.run_config.get("num-evaluate-clients", num_selected_clients))
    lr: float = context.run_config["learning-rate"]

    if num_supernodes <= 0:
        raise ValueError("num-supernodes deve ser maior que zero")
    if num_selected_clients <= 0:
        raise ValueError("num-selected-clients deve ser maior que zero")
    if num_evaluate_clients < 0:
        raise ValueError("num-evaluate-clients nao pode ser negativo")
    if num_selected_clients > num_supernodes:
        raise ValueError("num-selected-clients nao pode ser maior que num-supernodes")
    if num_evaluate_clients > num_supernodes:
        raise ValueError("num-evaluate-clients nao pode ser maior que num-supernodes")

    # Flower FedAvg recebe frações, mas o arquivo de configuração usa números absolutos.
    # Assim, o usuário configura quantidades fixas e o código calcula internamente.
    fraction_train: float = num_selected_clients / num_supernodes
    fraction_evaluate: float = 0.0 if num_evaluate_clients == 0 else num_evaluate_clients / num_supernodes
    log_dir: str = str(context.run_config.get("log-dir", "logs"))

    # Start each fl execution from a clean per-execution log.
    get_log_dir(log_dir).mkdir(parents=True, exist_ok=True)
    for execution_id in range(1, num_executions + 1):
        clear_execution_logs(execution_id=execution_id, log_dir=log_dir)

        # Load global model
        with Timer() as timer:
            global_model = Net()
            arrays = ArrayRecord(global_model.state_dict())
        write_log(
            execution_id=execution_id,
            server_round=0,
            node_id="server",
            stage="inicializa_modelo",
            elapsed_sec=timer.elapsed,
            details=(
                "Net()+ArrayRecord(global_model.state_dict()); "
                f"num_supernodes_config={num_supernodes}; "
                f"num_selected_clients_config={num_selected_clients}; "
                f"num_evaluate_clients_config={num_evaluate_clients}"
            ),
            log_dir=log_dir,
        )

        # Initialize FedAvg strategy.
        # Fixed-client validation using absolute values from pyproject.toml:
        # - num-supernodes controls how many simulated SuperNodes should exist.
        # - num-selected-clients controls how many clients are sampled per round.
        # The strategy API requires fractions, so they are calculated internally.
        strategy = TimedFedAvg(
            execution_id=execution_id,
            log_dir=log_dir,
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=num_selected_clients,
            # num_evaluate_clients=0 desativa a avaliação federada por clientes.
            min_evaluate_nodes=num_evaluate_clients,
            min_available_nodes=num_supernodes,
        )

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord(
                {
                    "lr": lr,
                    "execution_id": execution_id,
                    "server_round": 0,
                }
            ),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )

        # Save final model to disk
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_model_exec_{execution_id:03d}.pt")

    # Create compiled summaries after all executions.
    compile_summary(
        log_dir=log_dir,
        execution_ids=list(range(1, num_executions + 1)),
        warmup_rounds=warmup_rounds,
    )


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
