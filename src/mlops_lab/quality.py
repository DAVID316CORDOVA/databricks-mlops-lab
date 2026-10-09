"""Control de calidad de datos: no depende de Spark, solo de numeros (filas antes y despues)."""


class DataQualityError(Exception):
    """Los datos no cumplen el minimo de calidad: el job debe fallar (y avisar por correo)."""


def check_quality(rows_in: int, rows_out: int, max_drop_ratio: float = 0.5,
                  min_rows: int = 1000) -> dict:
    """Mide cuantas filas se descartaron al limpiar y falla si son demasiadas o quedan muy pocas.
    Sin esto, `dropna()` y los filtros eliminan datos en silencio y el job sale 'exitoso'."""
    dropped = 0.0 if rows_in == 0 else 1 - rows_out / rows_in
    problems = []
    if rows_out < min_rows:
        problems.append(f"solo {rows_out} filas utiles (minimo {min_rows})")
    if dropped > max_drop_ratio:
        problems.append(f"se descartaron {dropped:.1%} de las filas (maximo {max_drop_ratio:.0%})")
    if problems:
        raise DataQualityError("; ".join(problems))
    return {"rows_in": rows_in, "rows_out": rows_out, "dropped_ratio": round(dropped, 4)}