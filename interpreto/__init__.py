def get_version() -> str:
    """Returns the current version of the Inseq library."""
    try:
        import pkg_resources

        return pkg_resources.get_distribution("interpreto").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


__all__ = []
