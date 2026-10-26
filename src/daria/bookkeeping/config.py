import os, yaml
from importlib import resources

class Config:

    DEFAULT_CONFIG_PATH = str(resources.files('daria') / \
                              'bookkeeping/config_template.yaml')

    DEFAULT_USER_CONFIG_PATH = os.path.expandvars(\
            os.path.expandvars('$DARIA/config.yaml'))

    _DIRS_FROM_CONFIG_SETUP_SPECS = [
        ("conti_dir","conti_directory"), # continuum autos
        ("noise_dir","noise_directory"), # noise realizations
        ("inference_dir","inference_directory"), # inference outputs
        ("fig_dir","fig_directory"), # figures
    ]
    
    def __init__(self,user_config_path=None,auto_create_dirs=True):
        if user_config_path is None:
            user_config_path = self.DEFAULT_USER_CONFIG_PATH

        self.config_data = self.__load_config_data(user_config_path)
        self.serial = self.__get_serial_number()

        for attr, key in self._DIRS_FROM_CONFIG_SETUP_SPECS:
            setattr(self,attr,self.__get_formatted_dir_path(key))

        if auto_create_dirs:
            self.create_directories()

    def __load_config_data(self,user_config_path=None):
        default_path = self.DEFAULT_CONFIG_PATH
        try:
            with open(default_path,"r") as f:
                config_data = yaml.load(f,Loader=yaml.FullLoader)
        except Exception as e:
            err_str = f'Failed to load file {default_path}. Error: {e}'
            raise RuntimeError(err_str)

        if user_config_path is None:
            return config_data
        if os.path.exists(user_config_path):
            with open(user_config_path,"r") as f:
                user_config_data = yaml.load(f,Loader=yaml.FullLoader)
                config_data.update(user_config_data)
        elif user_config_path != default_path:
            print(f'User-specified path {user_config_path} does not exist')

        return config_data

    def __get_serial_number(self):
        return int(self.config_data.get("serial",0))

    def __get_formatted_dir_path(self,key):
        if key not in self.config_data:
            raise KeyError(f'Key {key} not in config data')
        dir_path = os.path.join(os.path.expandvars(self.get(key)),
                                f'{self.serial:04}')
        return dir_path
    
    def get(self,key):
        return self.config_data.get(key)

    def create_directories(self):
        for dir_attr,_ in self._DIRS_FROM_CONFIG_SETUP_SPECS:
            dir_path = getattr(self,dir_attr)
            if not os.path.exists(dir_path):
                print(f'Creating directory: {dir_path}')
                os.makedirs(dir_path,exist_ok=True)
