def get_version() -> str:
    """Returns the current version of the Inseq library."""
    import pkg_resources

    try:
        return pkg_resources.get_distribution("interpreto").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


__all__ = []
