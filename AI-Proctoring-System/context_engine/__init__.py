"""
Context Engine Package - AI Proctoring System

This package contains the minimal context analysis engine for the proctoring system.
"""

# Import simple components (minimal implementation)
try:
    from .simple_analyzer import SimpleAnalyzer
    from .simple_config import SimpleConfig
except ImportError as e:
    print(f"Warning: Context engine components not available: {e}")
    SimpleAnalyzer = None
    SimpleConfig = None

__all__ = [
    'SimpleAnalyzer',
    'SimpleConfig'
]