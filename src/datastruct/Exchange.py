import pickle
import requests
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class Exchange(ABC):
    """Base class for data exchange between objects."""
    
    def __init__(self) -> None:
        self._data: Optional[Any] = None

    @abstractmethod
    def populate(self, source: Any) -> None:
        """Populate data from source.
        
        Args:
            source: The source object to populate data from.
        """
        pass

    @property
    def data(self) -> Optional[Any]:
        """Get the underlying data object."""
        return self._data

    def save(self, filename: str) -> None:
        """Save the object to a file using pickle.
        
        Args:
            filename: Path to save the file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> Any:
        """Load an object from a file using pickle.
        
        Args:
            filename: Path to load the file from.
            
        Returns:
            The loaded object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def fetch(options: Dict[str, Any], url_or_name: str) -> Any:
        """Fetch data from a URL.
        
        Args:
            options: Dictionary of options (e.g., 'baseurl').
            url_or_name: URL or filename to fetch.
            
        Returns:
            Parsed JSON data.
        """
        url = url_or_name
        if not url.startswith('http'):
             url = options.get('baseurl', '') + url_or_name
             
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
