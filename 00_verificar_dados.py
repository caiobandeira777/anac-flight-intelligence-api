import polars as pl

df = pl.read_parquet("data/processed/anac_0_00.parquet")

print("=== SHAPE ===")
print(df.shape)

print("\n=== NULOS (%) ===")
print(df.null_count() / len(df) * 100)

print("\n=== TAXA OCUPACAO FORA DE 0-1 ===")
print(df.filter(
    (pl.col("taxa_ocupacao") > 1.0) | (pl.col("taxa_ocupacao") < 0.0)
).shape[0])

print("\n=== ASSENTOS ZERO OU NEGATIVO ===")
print(df.filter(pl.col("nr_assentos_ofertados") <= 0).shape[0])

print("\n=== DUPLICATAS ===")
print(df.is_duplicated().sum())

print("\n=== ESTATÍSTICAS BÁSICAS ===")
print(df.select([
    "taxa_ocupacao", "nr_assentos_ofertados",
    "km_distancia", "kg_bagagem_excesso"
]).describe())