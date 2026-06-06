# Quickstart PyTorch com logs de tempo e múltiplas execuções

Este projeto executa uma simulação Flower/PyTorch com logs por execução e resumos compilados.

## Configuração principal

No `pyproject.toml`, os valores agora usam quantidades absolutas, sem fração para selecionar clientes:

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

- `num-supernodes`: quantidade de SuperNodes simulados.
- `num-selected-clients`: quantidade fixa de clientes selecionados em cada rodada de treino.
- `num-evaluate-clients`: quantidade fixa de clientes usados na avaliação federada.

O Flower ainda recebe frações internamente na estratégia `FedAvg`, mas elas são calculadas automaticamente a partir desses valores. Assim, o arquivo de configuração fica em números absolutos.

## Importante sobre SuperNodes

Nas versões atuais do Flower, a quantidade de SuperNodes da simulação local também precisa existir na configuração da federação local. Para aplicar o valor do `pyproject.toml`, você pode rodar:

```powershell
python -m flwr federation simulation-config --num-supernodes 10 local
```

Depois execute:

```powershell
python -m flwr run . local --stream
```

Também incluí o script `run_fixed.ps1`, que lê `num-supernodes` do `pyproject.toml`, configura a simulação local e roda o Flower.

## Logs gerados

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
logs/compiled_without_warmup.csv
```

Os arquivos compilados incluem médias por etapa e `tempo_total_medio`.

## Informações técnicas sobre clientes e avaliação

Os parâmetros `num-selected-clients` e `num-evaluate-clients` representam fases diferentes do Flower.

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

- `num-supernodes`: total de clientes disponíveis na simulação. Por exemplo, `10` indica que a simulação terá 10 SuperNodes/clientes disponíveis.
- `num-selected-clients`: quantidade fixa de clientes usados no treinamento federado em cada rodada. Com `10`, o servidor seleciona 10 clientes, envia o modelo para esses 10 clientes, eles treinam localmente, devolvem os pesos para a cloud e a cloud agrega os 10 resultados.
- `num-evaluate-clients`: quantidade de clientes usados na avaliação federada. Essa etapa é diferente do treinamento: o servidor envia o modelo para os clientes avaliarem com dados locais de validação/teste, e os clientes retornam métricas como loss e acurácia.

Para validação dos tempos do ciclo de treinamento, recomenda-se usar:

```toml
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
```

Essa configuração mantém 10 clientes disponíveis e fixa a seleção de 10 clientes por rodada de treinamento. A avaliação federada fica desativada, evitando misturar o tempo de avaliação com os tempos principais que serão comparados com o modelo SPN.

As etapas principais registradas nos logs são:

```text
inicializa_modelo
seleciona_clientes
envia_para_clientes
treinamento_por_cliente
envio_cloud_por_cliente
agregacao_de_todos_clientes
```

A etapa `envia_para_clientes` inclui o envio do modelo, a execução dos clientes e o retorno das respostas ao servidor. Por isso, para evitar dupla contagem, o cálculo de tempo total médio nos arquivos compilados usa as etapas do servidor como referência principal.


## Correção sobre `num-evaluate-clients = 0`

A configuração `num-evaluate-clients = 0` é válida neste projeto e significa que a avaliação federada feita pelos clientes será desativada. Isso é útil quando o objetivo é medir apenas o ciclo principal de treinamento: seleção dos clientes, envio do modelo, treinamento local, retorno dos pesos e agregação no servidor.

O código permite `num-evaluate-clients = 0` e calcula internamente `fraction_evaluate = 0.0`. Assim, os 10 clientes continuam fixos no treinamento, enquanto a etapa extra de avaliação por clientes não entra nos tempos.

## Correção do cálculo de tempo total

Os arquivos compilados agora calculam o tempo total médio de duas formas:

- `tempo_total_medio_por_execucao_com_warmup` / `tempo_total_medio_por_execucao_sem_warmup`: média entre execuções da soma das rodadas consideradas.
- `tempo_total_medio_por_rodada_com_warmup` / `tempo_total_medio_por_rodada_sem_warmup`: média de duração das rodadas consideradas.

O total por rodada é calculado com:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

A etapa `treinamento_por_cliente` aparece no resumo como média por cliente, mas não é somada diretamente ao tempo total, porque os clientes executam em paralelo e esse tempo já está embutido em `envia_para_clientes`.

Algumas versões do Flower não expõem diretamente o ponto interno usado para registrar `envia_para_clientes`. Quando isso acontece, o código insere automaticamente no log de cada execução uma linha estimada para `envia_para_clientes`, calculada pelos timestamps do servidor:

```text
envia_para_clientes ~= início da agregação - fim da seleção dos clientes
```

Assim, os arquivos compilados deixam de apresentar um tempo total artificialmente baixo quando a etapa `envia_para_clientes` não aparece diretamente no log.


## Observação sobre o tempo total e os clientes

O arquivo `compiled_without_warmup.csv` apresenta dois totais principais:

- `tempo_total_medio_por_rodada_sem_warmup`: média do tempo de uma rodada de treinamento após remover as rodadas de warmup.
- `tempo_total_medio_por_execucao_sem_warmup`: média do tempo total de uma execução após remover as rodadas de warmup. Por exemplo, se `num-server-rounds = 5` e `warmup-rounds = 1`, esse total considera 4 rodadas por execução.

O tempo total por rodada é calculado como:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

A etapa `envia_para_clientes` já inclui o envio do modelo, o treinamento nos clientes, o retorno das respostas e overheads do Flower/Ray. Por isso, os tempos `treinamento_por_cliente` não são somados diretamente ao tempo total, pois os clientes executam em paralelo ou em ondas de paralelismo. Somar o treinamento de todos os clientes contaria o tempo como se eles treinassem de forma sequencial.

O projeto grava eventos em arquivos temporários individuais dentro de `logs/_events/` para evitar perda ou corrupção de linhas quando vários clientes escrevem ao mesmo tempo. Ao final, esses eventos são consolidados em `fl_timing_exec_001.csv`, `fl_timing_exec_002.csv` e assim por diante.


## Updated compiled summary behavior

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

The confidence interval is calculated using Student's t distribution over the total time of each execution. The total time per execution is computed from the server-side stages:

```text
seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes
```

The client-side stages are kept as stage-level averages, but they are not summed into the total because client execution is parallel/overlapped and already included in `envia_para_clientes`.



# APAGAR LOGS

taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe

Remove-Item -Recurse -Force logs