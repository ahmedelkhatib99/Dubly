

_pad        = "_"
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet

char_int_map = {c: i for i, c in enumerate(symbols)}
int_char_map = {i: c for i, c in enumerate(symbols)}


def text_to_int(text):
    return [char_int_map[c] for c in text]

def int_to_text(ints):
    return "".join([int_char_map[i] for i in ints])


