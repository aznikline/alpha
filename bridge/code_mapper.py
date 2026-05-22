"""StockCodeMapper — bidirectional stock code conversion between OpenAlpha and qmt formats.

OpenAlpha stores stock codes as 6-digit integers (e.g. 000001, 600000).
qmt stores stock codes with exchange suffix (e.g. 000001.SZ, 600000.SH).

Exchange rules:
- Shanghai (SH): codes >= 600000 (main board: 600xxx-603xxx, 605xxx, STAR board: 688xxx)
- Shenzhen (SZ): codes < 4000 (main board: 000xxx-003xxx) and 300xxx-301xxx (创业板)
- Beijing (BJ): codes 430xxx-830xxx (北交所) — future extension
"""
from __future__ import annotations


class StockCodeMapper:
    """Bidirectional mapping between OpenAlpha (integer/6-digit) and qmt (suffix) code formats.
    
    All methods are static — no instance state needed.
    Deterministic: same input always produces same output.
    """

    # Shanghai main board + STAR board (科创板)
    _SH_RANGES: list[tuple[int, int]] = [
        (600000, 690000),  # 600xxx-689xxx (main + 科创板 688xxx)
    ]

    # Shenzhen main board + 创业板
    _SZ_RANGES: list[tuple[int, int]] = [
        (0, 4000),          # 000xxx-003xxx (main board)
        (300000, 301000),   # 300xxx (创业板)
    ]

    # Beijing stock exchange (北交所) — future extension
    _BJ_RANGES: list[tuple[int, int]] = [
        (430000, 440000),   # 430xxx (old NEEQ)
        (830000, 840000),   # 830xxx (new BSE)
    ]

    @staticmethod
    def _code_in_ranges(code_int: int, ranges: list[tuple[int, int]]) -> bool:
        """Check if a code integer falls within any of the given ranges."""
        return any(lo <= code_int < hi for lo, hi in ranges)

    @staticmethod
    def to_qmt(code: int | str) -> str:
        """Convert OpenAlpha integer/6-digit code to qmt suffix format.
        
        Examples:
            1 → "000001.SZ"
            000001 → "000001.SZ"
            600000 → "600000.SH"
            300001 → "300001.SZ"  (创业板)
            688001 → "688001.SH"  (科创板)
        
        Args:
            code: Stock code as integer or 6-digit string (without suffix).
        
        Returns:
            qmt-format code with exchange suffix.
        
        Raises:
            ValueError: If code cannot be mapped to a known exchange.
        """
        code_int = int(str(code).replace(".SZ", "").replace(".SH", "").replace(".BJ", ""))
        # Already in qmt format? Return as-is.
        if isinstance(code, str) and "." in code:
            return code

        if StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._SH_RANGES):
            suffix = "SH"
        elif StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._SZ_RANGES):
            suffix = "SZ"
        elif StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._BJ_RANGES):
            suffix = "BJ"
        else:
            raise ValueError(
                f"Unknown exchange for code {code_int}. "
                f"Known ranges: SH={StockCodeMapper._SH_RANGES}, "
                f"SZ={StockCodeMapper._SZ_RANGES}, BJ={StockCodeMapper._BJ_RANGES}"
            )

        return f"{code_int:06d}.{suffix}"

    @staticmethod
    def to_openalpha(code: str) -> str:
        """Convert qmt suffix format to OpenAlpha 6-digit string.
        
        Examples:
            "000001.SZ" → "000001"
            "600000.SH" → "600000"
            "300001.SZ" → "300001"
        
        Note: Returns 6-digit string, not integer, to preserve leading zeros.
        
        Args:
            code: Stock code in qmt format (with suffix).
        
        Returns:
            6-digit string without suffix.
        """
        return code.split(".")[0]

    @staticmethod
    def to_int(code: str | int) -> int:
        """Convert any format to integer.
        
        Examples:
            "000001.SZ" → 1
            "600000.SH" → 600000
            1 → 1
        
        Args:
            code: Stock code in any format.
        
        Returns:
            Integer code value.
        """
        if isinstance(code, int):
            return code
        return int(code.split(".")[0])

    @staticmethod
    def batch_to_qmt(codes: list[int | str]) -> list[str]:
        """Batch conversion: OpenAlpha codes → qmt format.
        
        Args:
            codes: List of codes in OpenAlpha format.
        
        Returns:
            List of codes in qmt suffix format.
        """
        return [StockCodeMapper.to_qmt(c) for c in codes]

    @staticmethod
    def batch_to_openalpha(codes: list[str]) -> list[str]:
        """Batch conversion: qmt codes → OpenAlpha 6-digit strings.
        
        Args:
            codes: List of codes in qmt format.
        
        Returns:
            List of 6-digit strings without suffix.
        """
        return [StockCodeMapper.to_openalpha(c) for c in codes]

    @staticmethod
    def get_exchange(code: int | str) -> str:
        """Get exchange suffix for a stock code.
        
        Args:
            code: Stock code in any format.
        
        Returns:
            Exchange suffix: "SH", "SZ", or "BJ".
        
        Raises:
            ValueError: If exchange unknown.
        """
        code_int = int(str(code).split(".")[0])
        if StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._SH_RANGES):
            return "SH"
        elif StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._SZ_RANGES):
            return "SZ"
        elif StockCodeMapper._code_in_ranges(code_int, StockCodeMapper._BJ_RANGES):
            return "BJ"
        else:
            raise ValueError(f"Unknown exchange for code {code_int}")