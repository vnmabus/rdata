def write(fpath, data, format):
    if format == 'ascii':
        from .ascii import WriterASCII as Writer
        mode = 'w'
    elif format == 'xdr':
        from .xdr import WriterXDR as Writer
        mode = 'wb'
    else:
        raise ValueError(f'Unknown format: {format}')

    with open(fpath, mode) as f:
        w = Writer(f)
        w.write_all(data)

