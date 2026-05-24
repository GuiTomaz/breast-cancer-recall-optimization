# Breast Cancer Recall Optimization 

Este projeto foi desenvolvido como parte de um trabalho acadêmico de Machine Learning na **Universidade Federal Fluminense (UFF)**. O objetivo principal é construir e comparar modelos preditivos utilizando o dataset *Breast Cancer Wisconsin (Diagnostic)* para identificar tumores com foco na maximização da métrica de **Recall**, minimizando ao máximo a ocorrência de falsos negativos em diagnósticos médicos.

## Arquitetura e Engenharia de Dados Centralizada

Para garantir o rigor metodológico e evitar problemas crônicos de desenvolvimento em equipe (como conflitos de merge no Git e divergências de pré-processamento), este repositório utiliza uma **abordagem modular**:

1. **Pipeline Imutável (`pipeline.py`):** Toda a inteligência de carga, limpeza, divisão estratificada de dados (`test_size=0.2`, `stratify=y`, `random_state=42`) e padronização de escala com `StandardScaler` foi isolada em um script Python puro na raiz do projeto. Isso blinda o dataset contra o *Data Leakage* (vazamento de dados estatísticos do teste para o treino).
2. **Ambiente Padronizado (`.vscode/settings.json`):** Configuração embutida que força o Jupyter a rodar sempre a partir da raiz do projeto, garantindo automação total dos caminhos relativos e permitindo que qualquer integrante do grupo importe o pipeline com uma única linha de código.

---

## 📁 Estrutura do Repositório

```text
breast-cancer-recall-optimization/
│
├── .vscode/                         # Configurações de automação do ambiente
│   └── settings.json
│
├── data/                            # Dados originais imutáveis (repositório UCI)
│   ├── wdbc.data
│   └── wdbc.names
│
├── notebooks/                       # Desenvolvimento isolado de modelos (.ipynb)
│   ├── 00_analise_exploratoria.ipynb  # Carga, auditoria e engenharia inicial
│   └── 01_baseline_decision_tree.ipynb # Modelo de referência (Baseline)
│
├── outputs/                         # Artefatos exportados (Matrizes de Confusão)
│   └── matriz_decision_tree.png
│
├── pipeline.py                      # Script central do pipeline de dados
└── requirements.txt                 # Dependências do projeto para replicação
