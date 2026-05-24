# pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_pipeline():
    """
    Executa o pipeline completo de pré-processamento
    Retorna os conjuntos: X_train, X_test, y_train, y_test perfeitamente tratados.
    """
    # 1. Definição das colunas
    features_base = ["radius", "texture", "perimeter", "area", "smoothness",
                     "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]

    colunas = ["id", "diagnosis"]
    colunas += [f"{f}_mean" for f in features_base]
    colunas += [f"{f}_se" for f in features_base]
    colunas += [f"{f}_worst" for f in features_base]

    # 2. Carga dos dados
    df = pd.read_csv("data/wdbc.data", header=None, names=colunas)

    # 3. Limpeza (Remoção do ID)
    clean_df = df.drop(columns=['id'])

    # 4. Separação de Variáveis e Mapeamento do Alvo
    X = clean_df.drop(columns=['diagnosis'])
    y = clean_df['diagnosis'].map({'M': 1, 'B': 0})

    # 5. Divisão Estratificada (Treino/Teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Padronização (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Evitando Data Leakage usando apenas transform no teste
    X_test_scaled = scaler.transform(X_test)

    # 7. Conversão de volta para DataFrame (Mantendo os nomes das colunas)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # Aqui usamos X_train.columns pois as colunas de teste são idênticas
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test