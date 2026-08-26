"""Command-line entry point for prebuilding the C kernels."""
from ._c.build import KERNELS, ensure_lib, main

__all__ = ["KERNELS", "ensure_lib", "main"]

if __name__ == "__main__":
    main()
