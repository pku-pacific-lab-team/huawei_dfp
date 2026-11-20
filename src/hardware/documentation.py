"""Optional JSON maintenance notes, separate from validated model inputs."""


def without_documentation(config):
    """Ignore only the reserved root annotation, not arbitrary unknown fields.

    Keep the caller's dictionary intact. Runtime reference IDs exclude these
    notes; the source-audit checksum still covers the complete JSON document.
    """
    if not isinstance(config, dict):
        raise ValueError("hardware configuration must be an object")
    if "_documentation" in config and not isinstance(config["_documentation"], dict):
        raise ValueError("hardware._documentation must be an object")
    return {key: value for key, value in config.items() if key != "_documentation"}
