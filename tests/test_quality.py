import pytest

from mlops_lab.quality import DataQualityError, check_quality


def test_calidad_ok():
    stats = check_quality(1000, 900, max_drop_ratio=0.2, min_rows=100)
    assert stats["rows_out"] == 900 and stats["dropped_ratio"] == 0.1


def test_demasiado_descarte_falla():
    with pytest.raises(DataQualityError):
        check_quality(1000, 500, max_drop_ratio=0.2, min_rows=100)


def test_pocas_filas_falla():
    with pytest.raises(DataQualityError):
        check_quality(50, 50, max_drop_ratio=0.2, min_rows=100)