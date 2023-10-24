import time

def timing(f):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Écrire les résultats dans un fichier
        with open("execution_times.txt", "a") as file:
            file.write(f"{f.__name__},{end_time - start_time}\n")
        
        return result
    return wrap
