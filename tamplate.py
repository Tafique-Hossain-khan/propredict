from pathlib import Path
import os
import logging
logging.basicConfig(filename='logginginfo.log',level=logging.INFO,format='%(asctime)s %(message)s')
project_name = "laptop_price"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/utils.py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/data_injection.py",
    
    
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/traning_pipeline .py",
    f"src/{project_name}/pipeline/prediction_pipeline .py",
    f'notebook/{project_name}/main_notbook.ipynb',
    f'notebook/{project_name}/rogh_notbook.ipynb',
    
    
   
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "application.py",
    "requirements.txt",
    "setup.py",
    
    


]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")

