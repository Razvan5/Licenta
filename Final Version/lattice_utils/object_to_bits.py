import os


def text_to_bits(text):

    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return list(map(int, bits.zfill(8 * ((len(bits) + 7) // 8))))


def text_from_bits(bits):

    n = int(''.join(map(str, bits)), 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()


def unique_filename(output_filename, file_extension):
    n = ''
    while os.path.exists(f'{output_filename}{n}{file_extension}'):
        if isinstance(n, str):
            n = -1
        n += 1
    return f'{output_filename}{n}{file_extension}'