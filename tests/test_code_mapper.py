"""Tests for bridge.code_mapper — StockCodeMapper."""
import pytest

from bridge.code_mapper import StockCodeMapper


class TestToQmt:
    def test_sz_main_board(self):
        assert StockCodeMapper.to_qmt(1) == "000001.SZ"
        assert StockCodeMapper.to_qmt("000001") == "000001.SZ"
        assert StockCodeMapper.to_qmt("000002") == "000002.SZ"
    
    def test_sz_gem_board(self):
        assert StockCodeMapper.to_qmt(300001) == "300001.SZ"
        assert StockCodeMapper.to_qmt(300999) == "300999.SZ"
    
    def test_sh_main_board(self):
        assert StockCodeMapper.to_qmt(600000) == "600000.SH"
        assert StockCodeMapper.to_qmt(603000) == "603000.SH"
    
    def test_sh_star_board(self):
        assert StockCodeMapper.to_qmt(688001) == "688001.SH"
    
    def test_already_has_suffix(self):
        # Idempotent: already in qmt format → return unchanged
        assert StockCodeMapper.to_qmt("000001.SZ") == "000001.SZ"
        assert StockCodeMapper.to_qmt("600000.SH") == "600000.SH"
    
    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            StockCodeMapper.to_qmt(500000)  # Not in any range
    
    def test_integer_input(self):
        assert StockCodeMapper.to_qmt(1) == "000001.SZ"
        assert StockCodeMapper.to_qmt(600000) == "600000.SH"


class TestToOpenalpha:
    def test_strip_suffix(self):
        assert StockCodeMapper.to_openalpha("000001.SZ") == "000001"
        assert StockCodeMapper.to_openalpha("600000.SH") == "600000"
    
    def test_preserves_leading_zeros(self):
        assert StockCodeMapper.to_openalpha("000001.SZ") == "000001"
        assert StockCodeMapper.to_openalpha("000002.SZ") == "000002"


class TestToInt:
    def test_from_qmt_format(self):
        assert StockCodeMapper.to_int("000001.SZ") == 1
        assert StockCodeMapper.to_int("600000.SH") == 600000
    
    def test_from_integer(self):
        assert StockCodeMapper.to_int(1) == 1
        assert StockCodeMapper.to_int(600000) == 600000


class TestBatchConversion:
    def test_batch_to_qmt(self):
        codes = [1, 600000, 300001]
        result = StockCodeMapper.batch_to_qmt(codes)
        assert result == ["000001.SZ", "600000.SH", "300001.SZ"]
    
    def test_batch_to_openalpha(self):
        codes = ["000001.SZ", "600000.SH", "300001.SZ"]
        result = StockCodeMapper.batch_to_openalpha(codes)
        assert result == ["000001", "600000", "300001"]


class TestGetExchange:
    def test_shanghai(self):
        assert StockCodeMapper.get_exchange(600000) == "SH"
        assert StockCodeMapper.get_exchange(688001) == "SH"
    
    def test_shenzhen(self):
        assert StockCodeMapper.get_exchange(1) == "SZ"
        assert StockCodeMapper.get_exchange(300001) == "SZ"
    
    def test_beijing(self):
        assert StockCodeMapper.get_exchange(430001) == "BJ"
        assert StockCodeMapper.get_exchange(830001) == "BJ"
    
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            StockCodeMapper.get_exchange(500000)