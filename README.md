# Quickstart PyTorch with Instrumented Timing Logs and Multiple Executions

This project runs an instrumented Flower/PyTorch simulation that records timing information for the main stages of the federated learning workflow. It is useful for anyone who wants to analyze how much time is spent in each part of a Flower execution, such as client selection, model transmission, local training, client response, and server aggregation.

The project also supports multiple executions and automatically generates compiled summaries with average times and confidence intervals.

## Main Configuration

In `pyproject.toml`, the configuration uses absolute quantities instead of fractions for client selection:

```toml
[tool.flwr.app.config]
num-server-rounds = 5
num-executions = 3
warmup-rounds = 1
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
local-epochs = 1
learning-rate = 0.1
batch-size = 32
log-dir = "logs"
```

* `num-server-rounds`: number of federated learning rounds.
* `num-executions`: number of independent executions of the simulation.
* `warmup-rounds`: number of initial rounds used as warmup.
* `num-supernodes`: number of simulated SuperNodes/clients.
* `num-selected-clients`: fixed number of clients selected in each training round.
* `num-evaluate-clients`: fixed number of clients used in federated evaluation.
* `local-epochs`: number of local training epochs per client.
* `learning-rate`: learning rate used during training.
* `batch-size`: batch size used by the clients.
* `log-dir`: directory where the timing logs are stored.

Flower still receives fractions internally in the `FedAvg` strategy, but these fractions are automatically calculated from the absolute values defined in the configuration. This makes the configuration easier to read and adjust.

## Important Note About SuperNodes

In current Flower versions, the number of SuperNodes used in the local simulation must also be defined in the local federation configuration. To apply the value from `pyproject.toml`, run:

```powershell
python -m flwr federation simulation-config --num-supernodes 10 local
```

Then execute:

```powershell
python -m flwr run . local --stream
```

The script `run_fixed.ps1` is also included. It reads `num-supernodes` from `pyproject.toml`, configures the local simulation, and runs Flower.

## Generated Logs

After running the simulation, the project generates one timing log for each execution and a compiled summary file:

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
```

Each `fl_timing_exec_XXX.csv` file contains the timing events recorded during one execution.

The `compiled_with_warmup.csv` file contains the compiled timing summary across all executions.

If an old `compiled_without_warmup.csv` file exists from a previous run, the code tries to remove it, since the current version only generates `compiled_with_warmup.csv`.

## Instrumented Flower Stages

The main stages recorded in the logs are:

```text
inicializa_modelo
seleciona_clientes
envia_para_clientes
treinamento_por_cliente
envio_cloud_por_cliente
agregacao_de_todos_clientes
```

These stages allow users to inspect the main timing components of a Flower/PyTorch federated learning round.

* `inicializa_modelo`: time spent initializing the model.
* `seleciona_clientes`: time spent selecting clients for the current round.
* `envia_para_clientes`: time covering model transmission, client execution, and the return of client responses to the server.
* `treinamento_por_cliente`: local training time measured at the client side.
* `envio_cloud_por_cliente`: time related to sending the client update back to the cloud/server.
* `agregacao_de_todos_clientes`: time spent aggregating all selected client updates at the server.

The stage names are kept in Portuguese because they are directly used in the generated CSV files.

## Technical Information About Clients and Evaluation

The parameters `num-selected-clients` and `num-evaluate-clients` represent different phases of Flower.

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

* `num-supernodes`: total number of clients available in the simulation. For example, `10` means that the simulation will have 10 available SuperNodes/clients.
* `num-selected-clients`: fixed number of clients used in federated training in each round. With `10`, the server selects 10 clients, sends the model to them, the clients train locally, return their updates, and the server aggregates the results.
* `num-evaluate-clients`: number of clients used in federated evaluation. This stage is different from training. The server sends the model to clients so they can evaluate it using local validation/test data, and the clients return metrics such as loss and accuracy.

For measuring only the main training cycle, the recommended configuration is:

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

This keeps 10 clients available, fixes the selection of 10 clients per training round, and disables federated client-side evaluation. This avoids mixing evaluation time with the main training timing measurements.

## About `num-evaluate-clients = 0`

The configuration `num-evaluate-clients = 0` is valid in this project. It means that client-side federated evaluation is disabled.

This is useful when the goal is to measure only the main training cycle:

```text
client selection
model transmission
local training
client update return
server aggregation
```

Internally, the code allows `num-evaluate-clients = 0` and calculates `fraction_evaluate = 0.0`. Therefore, the selected clients remain fixed for training, while the additional evaluation stage is not included in the timing measurements.

## Total Time Calculation

The compiled summary includes total timing metrics based on the server-side stages. The total time per round is calculated as:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

The `treinamento_por_cliente` stage appears in the summary as an average per client, but it is not directly summed into the total time. This is because clients can execute in parallel or in overlapping waves, and their execution time is already embedded in `envia_para_clientes`.

Summing the training time of all clients directly would count the execution as if all clients trained sequentially, which does not represent how Flower simulations usually execute.

## Estimated `envia_para_clientes`

Some Flower versions do not directly expose the internal point used to record `envia_para_clientes`. When this happens, the code automatically inserts an estimated `envia_para_clientes` line into each execution log.

The estimate is calculated from server-side timestamps:

```text
envia_para_clientes ~= aggregation start - client selection end
```

This avoids generating an artificially low total time when the `envia_para_clientes` stage does not appear directly in the log.

## Compiled Summary Behavior

The project generates only one compiled summary file:

```text
logs/compiled_with_warmup.csv
```

The row `tempo_total_medio_por_execucao_com_warmup` includes the 95% confidence interval computed across the number of executions configured in `num-executions`:

```text
ci95_low_sec
ci95_high_sec
```

The confidence interval is calculated using Student's t-distribution over the total time of each execution.

The total time per execution is computed from the server-side stages:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

Client-side stages are kept as stage-level averages, but they are not directly summed into the total because client execution is parallel or overlapped and already included in `envia_para_clientes`.

## Temporary Event Files

The project writes timing events to individual temporary files inside:

```text
logs/_events/
```

This avoids line loss or file corruption when multiple clients write timing events at the same time.

At the end of each execution, these temporary events are consolidated into files such as:

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
```

## Cleaning Logs and Processes on Windows

To stop running processes and remove previous logs, use:

```powershell
taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe

Remove-Item -Recurse -Force logs
```

## Notes

This version is intended to provide a practical instrumented Flower/PyTorch workflow. It can be used to analyze execution time, compare configurations, inspect training behavior, and collect timing data from different stages of a federated learning simulation.
