def write(fpath, data, *, format, compression='gzip'):
    if format == 'ascii':
        from .ascii import WriterASCII as Writer
        mode = 'w'
    elif format == 'xdr':
        from .xdr import WriterXDR as Writer
        mode = 'wb'
    else:
        raise ValueError(f'Unknown format: {format}')

    if compression == 'gzip':
        from gzip import open
    elif compression == 'bzip2':
        from bz2 import open
    elif compression == 'xz':
        from lzma import open
    else:
        assert compression is None
        import builtins
        open = builtins.open

    with open(fpath, mode) as f:
        w = Writer(f)
        w.write_all(data)

