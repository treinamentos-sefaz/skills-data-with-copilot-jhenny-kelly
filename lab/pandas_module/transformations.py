"""Módulo 2 — Manipulação de DataFrames com pandas.

Complete cada função com o apoio do GitHub Copilot. Prefira operações
vetorizadas e evite alterar o DataFrame original (use ``.copy()``).

Valide com:  ``python lab/check.py --module pandas``
"""
from __future__ import annotations

import pandas as pd


def load_wines(path: str) -> pd.DataFrame:
    """Leia os dados de vinhos de um arquivo CSV.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os dados lidos.
    """
    # Prompt sugerido: "Leia o arquivo CSV em path com pandas e retorne o DataFrame."
    return pd.read_csv(path)


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Converta avaliações para números e remova valores inválidos ou nulos.

    Args:
        df: DataFrame com a coluna ``rating``.

    Returns:
        Novo DataFrame com avaliações numéricas não nulas.
    """
    result = df.copy()
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    return result.dropna(subset=["rating"])


def top_varieties_by_avg_rating(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Calcule as variedades com as maiores médias de avaliação.

    Args:
        df: DataFrame com as colunas ``variety`` e ``rating`` numérico.
        n: Número máximo de variedades retornadas.

    Returns:
        DataFrame com ``variety`` e ``avg_rating``, ordenado por média decrescente.
    """
    return (
        df.copy().groupby("variety", as_index=False)["rating"]
        .mean()
        .rename(columns={"rating": "avg_rating"})
        .sort_values("avg_rating", ascending=False)
        .head(n)
    )


def add_rating_category(df: pd.DataFrame) -> pd.DataFrame:
    """Classifique avaliações em baixa, média ou alta.

    Args:
        df: DataFrame com a coluna ``rating`` numérico.

    Returns:
        Cópia com ``rating_category``: "Baixa" para valores abaixo de 85,
        "Média" de 85 a 90 inclusive e "Alta" para valores acima de 90.
    """
    result = df.copy()
    result["rating_category"] = pd.cut(
        result["rating"],
        bins=[-float("inf"), 85, 90, float("inf")],
        labels=["Baixa", "Média", "Alta"],
        right=False,
    )
    result.loc[result["rating"].eq(90), "rating_category"] = "Média"
    return result


def enrich_with_country(df: pd.DataFrame) -> pd.DataFrame:
    """Extraia o último trecho de ``region`` separado por vírgula.

    Args:
        df: DataFrame com a coluna textual ``region``.

    Returns:
        Cópia com a coluna ``country``, sem espaços nas extremidades.
        Por exemplo, "Mendocino, California" resulta em "California".
    """
    result = df.copy()
    result["country"] = result["region"].str.split(",").str[-1].str.strip()
    return result
