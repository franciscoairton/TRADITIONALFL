# Quickstart PyTorch with Timing Logs and Multiple Executions

This project runs a Flower/PyTorch simulation with logs for each execution and compiled summaries.

## Main Configuration

In `pyproject.toml`, the values now use absolute quantities, without fractions for client selection:

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

* `num-supernodes`: number of simulated SuperNodes.
* `num-selected-clients`: fixed number of clients selected in each training round.
* `num-evaluate-clients`: fixed number of clients used in federated evaluation.

Flower still receives fractions internally in the `FedAvg` strategy, but they are automatically calculated from these values. This keeps the configuration file based on absolute numbers.

## Important Note About SuperNodes

In current Flower versions, the number of SuperNodes used in the local simulation must also exist in the local federation configuration. To apply the value from `pyproject.toml`, run:

```powershell
python -m flwr federation simulation-config --num-supernodes 10 local
```

Then execute:

```powershell
python -m flwr run . local --stream
```

The script `run_fixed.ps1` was also included. It reads `num-supernodes` from `pyproject.toml`, configures the local simulation, and runs Flower.

## Generated Logs

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
logs/compiled_without_warmup.csv
```

The compiled files include stage averages and `tempo_total_medio`.

## Technical Information About Clients and Evaluation

The parameters `num-selected-clients` and `num-evaluate-clients` represent different Flower phases.

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

* `num-supernodes`: total number of clients available in the simulation. For example, `10` means that the simulation will have 10 available SuperNodes/clients.
* `num-selected-clients`: fixed number of clients used in federated training in each round. With `10`, the server selects 10 clients, sends the model to these 10 clients, they train locally, return the weights to the cloud, and the cloud aggregates the 10 results.
* `num-evaluate-clients`: number of clients used in federated evaluation. This stage is different from training: the server sends the model to clients so they can evaluate it using local validation/test data, and the clients return metrics such as loss and accuracy.

For validating the training cycle times, the recommended configuration is:

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

This configuration keeps 10 clients available and fixes the selection of 10 clients per training round. Federated evaluation remains disabled, avoiding the inclusion of evaluation time in the main times that will be compared with the SPN model.

The main stages recorded in the logs are:

```text
inicializa_modelo
seleciona_clientes
envia_para_clientes
treinamento_por_cliente
envio_cloud_por_cliente
agregacao_de_todos_clientes
```

The `envia_para_clientes` stage includes model transmission, client execution, and the return of client responses to the server. Therefore, to avoid double counting, the average total time calculation in the compiled files uses the server-side stages as the main reference.

## Correction About `num-evaluate-clients = 0`

The configuration `num-evaluate-clients = 0` is valid in this project and means that client-side federated evaluation will be disabled. This is useful when the goal is to measure only the main training cycle: client selection, model transmission, local training, return of weights, and server aggregation.

The code allows `num-evaluate-clients = 0` and internally calculates `fraction_evaluate = 0.0`. Thus, the 10 clients remain fixed for training, while the additional client evaluation stage is not included in the timing measurements.

## Correction of the Total Time Calculation

The compiled files now calculate the average total time in two ways:

* `tempo_total_medio_por_execucao_com_warmup` / `tempo_total_medio_por_execucao_sem_warmup`: average across executions of the sum of the considered rounds.
* `tempo_total_medio_por_rodada_com_warmup` / `tempo_total_medio_por_rodada_sem_warmup`: average duration of the considered rounds.

The total per round is calculated as:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

The `treinamento_por_cliente` stage appears in the summary as an average per client, but it is not directly summed into the total time because clients execute in parallel and this time is already embedded in `envia_para_clientes`.

Some Flower versions do not directly expose the internal point used to record `envia_para_clientes`. When this happens, the code automatically inserts an estimated `envia_para_clientes` line into the log of each execution, calculated from server-side timestamps:

```text
envia_para_clientes ~= aggregation start - client selection end
```

This prevents the compiled files from showing an artificially low total time when the `envia_para_clientes` stage does not appear directly in the log.

## Note About Total Time and Clients

The `compiled_without_warmup.csv` file presents two main totals:

* `tempo_total_medio_por_rodada_sem_warmup`: average time of one training round after removing the warmup rounds.
* `tempo_total_medio_por_execucao_sem_warmup`: average total time of one execution after removing the warmup rounds. For example, if `num-server-rounds = 5` and `warmup-rounds = 1`, this total considers 4 rounds per execution.

The total time per round is calculated as:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

The `envia_para_clientes` stage already includes model transmission, client training, response return, and Flower/Ray overheads. Therefore, the `treinamento_por_cliente` times are not directly summed into the total time, because clients execute in parallel or in waves of parallelism. Summing the training time of all clients would count the time as if they trained sequentially.

The project writes events to individual temporary files inside `logs/_events/` to avoid line loss or corruption when multiple clients write at the same time. At the end, these events are consolidated into `fl_timing_exec_001.csv`, `fl_timing_exec_002.csv`, and so on.

## Updated Compiled Summary Behavior

The project now generates only one compiled summary file:

```text
logs/compiled_with_warmup.csv
```

The previous `compiled_without_warmup.csv` file is no longer generated. If an old file exists from a previous run, the code tries to remove it.

The row `tempo_total_medio_por_execucao_com_warmup` includes the 95% confidence interval computed across the N executions configured in `num-executions`:

```text
ci95_low_sec
ci95_high_sec
```

The confidence interval is calculated using Student's t-distribution over the total time of each execution. The total time per execution is computed from the server-side stages:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

The client-side stages are kept as stage-level averages, but they are not summed into the total because client execution is parallel/overlapped and already included in `envia_para_clientes`.

# DELETE LOGS

```powershell
taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe

Remove-Item -Recurse -Force logs
```
