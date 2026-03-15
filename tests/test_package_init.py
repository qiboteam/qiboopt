import importlib
import importlib.metadata as im


def test_qiboopt_version_fallback(monkeypatch):
    import qiboopt

    original_version = qiboopt.__version__
    original_metadata_version = im.version

    monkeypatch.setattr(
        im,
        "version",
        lambda _package: (_ for _ in ()).throw(RuntimeError("missing metadata")),
    )
    importlib.reload(qiboopt)

    assert qiboopt.__version__ == "0.0.1"

    monkeypatch.setattr(im, "version", original_metadata_version)
    importlib.reload(qiboopt)
    assert qiboopt.__version__ == original_version
