from __future__ import annotations


def prefer_certifi_default_context() -> None:
    """Force default SSL contexts to use certifi when Windows cert store is broken."""
    try:
        import certifi
    except ImportError:  # pragma: no cover
        return

    import ssl

    original_create_default_context = ssl.create_default_context
    cafile = certifi.where()

    def create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        *,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        return original_create_default_context(
            purpose=purpose,
            cafile=cafile or (None if cadata else cafile_path),
            capath=capath,
            cadata=cadata,
        )

    cafile_path = cafile
    ssl.create_default_context = create_default_context
