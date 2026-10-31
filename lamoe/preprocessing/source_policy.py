"""Source-neutral limits supplied explicitly by a local token manifest."""


def pass_limits(manifest, split):
    sources = manifest["sources"]
    configured = manifest.get("source_max_passes", {})
    if (
        not isinstance(configured, dict)
        or set(configured) - set(sources)
        or any(type(v) is not int or v < 1 for v in configured.values())
    ):
        raise ValueError(
            "source_max_passes must map known sources to positive integers"
        )
    return {
        name: configured.get(name, 1) if split == "train" else 1 for name in sources
    }


def continuation_policy(metadata, source_names):
    """Fail closed when a private continuation's source policy is absent."""
    if metadata is None:
        return frozenset(), 0
    if not isinstance(metadata, dict):
        raise ValueError("Explicit continuation metadata is required")
    introduced = metadata.get("introduced_sources")
    threshold = metadata.get("introduced_sources_start_tokens")
    if (
        not isinstance(introduced, list)
        or not introduced
        or any(not isinstance(name, str) for name in introduced)
        or len(set(introduced)) != len(introduced)
        or set(introduced) - set(source_names)
        or type(threshold) is not int
        or threshold < 0
    ):
        raise ValueError(
            "Explicit continuation sources and start-token threshold required"
        )
    return frozenset(introduced), threshold
