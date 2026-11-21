"""vc-reader-lite: a pure-Python stand-in for the parts of villa's `vc_reader` used by
`python -m vesuvius.ink_detection.inference.infer` (flat ink inference).

Install it *instead of* the native vc_reader when you cannot compile it (e.g. Windows
without the Visual Studio toolchain, see villa PR #2533 for the native Windows build).
Covered: reader.open_node / load_basic_auth / ZarrWriter.create, transforms.Normalize
(robust, percentile, fixed, zscore, minmax; unmasked), images.read / images.TiffWriter.
Anything else raises ImportError/AttributeError so it fails loudly instead of silently.
"""
LITE = True
__version__ = "0.0.1-lite"
