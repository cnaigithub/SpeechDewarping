from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def _symbols_to_sequence(symbols, clean_symbols=False):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)] if clean_symbols else [_symbol_to_id[s] for s in symbols]

def _should_keep_symbol(s):
    return s in _symbol_to_id# and s is not '_' and s is not '~'
