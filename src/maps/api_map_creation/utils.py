import time
import os

BUILD_DIR = os.path.join(os.path.dirname(__file__), '../build/API/')

def timing(f):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        file_path = os.path.join(BUILD_DIR, "execution_times.txt")
        # si le fichier n'existe pas, on le crée
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("")

        with open(file_path, "a") as file:
            file.write(f"{f.__name__},{end_time - start_time}\n")
        
        return result
    return wrap
