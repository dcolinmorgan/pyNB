class Exchange:
    """Base class for data exchange between objects."""
    
    def __init__(self):
        self._data = None  # Placeholder for subclass-specific data

    def populate(self, source):
        """Abstract method to populate data; override in subclasses."""
        raise NotImplementedError("Subclasses must implement populate")

    @property
    def data(self):
        return self._data
