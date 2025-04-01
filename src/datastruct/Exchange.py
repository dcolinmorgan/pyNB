import os
import json
import numpy as np
from pathlib import Path

class Exchange:
    """Base class for data exchange operations in GeneSPIDER."""

    def __init__(self):
        pass  # No specific initialization needed

    def populate(self, input):
        """Populate object attributes with matching fields from input."""
        if not isinstance(input, dict):
            raise ValueError("Input must be a dictionary-like structure")
        for key, value in input.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save(self, savepath=None, fending='.mat', **kwargs):
        """
        Save the object to a file in the specified format.

        Args:
            savepath (str, optional): Directory path to save the file. Defaults to current directory.
            fending (str, optional): File extension ('.mat', '.json', '.xml', '.ubj'). Defaults to '.mat'.
            **kwargs: Additional arguments passed to the save function.
        """
        if savepath is None:
            savepath = os.getcwd()
        if not os.path.isdir(savepath):
            raise FileNotFoundError(f"Save path does not exist: {savepath}")
        if not isinstance(fending, str):
            raise ValueError("File extension must be a string")

        # Convert object to dictionary
        obj_data = vars(self)

        # Determine filename based on object type
        if isinstance(self, Dataset):
            name = obj_data.get('dataset', 'unnamed_dataset')
            savevar = 'obj_data'
        elif isinstance(self, Network):  # Placeholder for Network class
            name = obj_data.get('network', 'unnamed_network')
            savevar = 'obj_data'
        else:
            print("Warning: Unknown object type")
            name = 'unknown_datatype'
            savevar = 'obj_data'

        filepath = os.path.join(savepath, f"{name}{fending}")

        # Save based on file extension
        if fending == '.json':
            with open(filepath, 'w') as f:
                json.dump({savevar: obj_data}, f, default=self._json_serializer, **kwargs)
        elif fending == '.mat':
            try:
                from scipy.io import savemat
                savemat(filepath, {savevar: obj_data}, **kwargs)
            except ImportError:
                raise ImportError("Saving .mat files requires scipy.io")
        elif fending == '.xml':
            raise NotImplementedError("XML saving is not implemented. Requires an external library.")
        elif fending.startswith('.ubj'):
            raise NotImplementedError("UBJ saving is not implemented. Requires an external library.")
        else:
            raise ValueError(f"Unknown file extension: {fending}")

    def _json_serializer(self, obj):
        """Helper to serialize NumPy arrays and other objects to JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.float_)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    @staticmethod
    def load(*args):
        """
        Load a dataset/network file into an object.

        Args:
            *args: Path (str), or (path, file) tuple. If numeric, selects from file list.

        Returns:
            Exchange subclass instance or list of files if no specific file is given.
        """
        lpath = os.getcwd()
        lfile = None

        if len(args) == 1:
            if isinstance(args[0], (int, float)):
                lfile = int(args[0])
            elif os.path.isfile(args[0]):
                lpath, lfile = os.path.split(args[0])
            elif os.path.isdir(args[0]):
                lpath = args[0]
            else:
                raise ValueError("Unknown path or file")
        elif len(args) == 2:
            lpath = args[0]
            if not os.path.isdir(lpath):
                raise FileNotFoundError(f"Unknown path: {lpath}")
            if isinstance(args[1], (int, float)):
                lfile = int(args[1])
            elif os.path.isfile(os.path.join(lpath, args[1])):
                lfile = args[1]
            else:
                raise ValueError("Unknown file")
        elif len(args) > 2:
            raise ValueError("Too many input arguments")

        # If lfile is an index, list files and select
        if isinstance(lfile, int):
            files = [f for f in os.listdir(lpath) if f.endswith(('.mat', '.json', '.xml', '.ubj'))]
            if not files:
                print(f"Warning: No data files in directory: {lpath}")
                return None
            if lfile < 1 or lfile > len(files):
                raise IndexError("File index out of range")
            lfile = files[lfile - 1]
        elif lfile is None:
            files = [f for f in os.listdir(lpath) if f.endswith(('.mat', '.json', '.xml', '.ubj'))]
            return files

        fetchfile = os.path.join(lpath, lfile)
        _, ext = os.path.splitext(fetchfile)

        # Load based on file extension
        if ext == '.mat':
            from scipy.io import loadmat
            obj_data = loadmat(fetchfile)
            obj_data = obj_data.get('obj_data', obj_data.get('dataset', obj_data.get('network')))
        elif ext == '.json':
            with open(fetchfile, 'r') as f:
                obj_data = json.load(f)
                obj_data = next(iter(obj_data.values()))  # Get first value (assuming one key)
        elif ext == '.xml':
            raise NotImplementedError("XML loading is not implemented. Requires an external library.")
        elif ext.startswith('.ubj'):
            raise NotImplementedError("UBJ loading is not implemented. Requires an external library.")
        else:
            raise ValueError(f"Unknown file extension: {ext}")

        # Instantiate appropriate class
        if 'Y' in obj_data:
            obj = Dataset()
        elif 'A' in obj_data:
            obj = Network()  # Placeholder for Network class
        else:
            raise ValueError("No compatible class for the data file")
        
        obj.populate(obj_data)
        return obj

    @staticmethod
    def fetch(options=None, *args):
        """
        Fetch data from a remote repository (e.g., Bitbucket).

        Args:
            options (dict, optional): Configuration options (directurl, baseurl, version, etc.).
            *args: URL or name-value pairs.

        Returns:
            Fetched data or list of files.
        """
        import requests

        default_options = {
            'directurl': '',
            'baseurl': 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/',
            'version': 'master',
            'N': 10,
            'name': 'Nordling-ID1446937-D20150825-N10-E15-SNR3291-IDY15968',
            'filetype': '.json'
        }
        options = {**default_options, **(options or {})}

        if len(args) == 1:
            directurl, name, filetype = os.path.split(args[0])
            options.update({'directurl': directurl, 'name': name, 'filetype': filetype})
        elif len(args) > 1:
            if len(args) % 2 != 0:
                raise ValueError("fetch needs propertyName/propertyValue pairs")
            for key, value in zip(args[::2], args[1::2]):
                if key in options:
                    options[key] = value
                else:
                    raise ValueError(f"{key} is not a recognized parameter name")

        name, ext = os.path.splitext(options['name'])
        if ext:
            options['filetype'] = ext
            options['name'] = name

        urls = [
            os.path.join(options['directurl'], f"{options['name']}{options['filetype']}"),
            os.path.join(options['baseurl'], options['directurl'], f"{options['name']}{options['filetype']}"),
            os.path.join(options['baseurl'], options['version'], options['directurl'], f"{options['name']}{options['filetype']}")
        ]

        for url in urls:
            try:
                response = requests.get(url.replace('\\', '/'))
                response.raise_for_status()
                if options['filetype'] == '.json':
                    obj_data = response.json()
                elif options['filetype'] == '.mat':
                    raise NotImplementedError("MAT fetching not implemented directly")
                else:
                    obj_data = response.text.splitlines()
                    return obj_data
                return obj_data.get('obj_data', obj_data)
            except requests.RequestException:
                continue
        raise ValueError(f"Could not fetch data from URLs: {urls}")
