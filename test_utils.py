import threading
import time
import functools

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            completed = [False]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                    completed[0] = True
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            start_time = time.time()
            dots = 0
            while time.time() - start_time < seconds and not completed[0]:
                time.sleep(0.1)
                dots = (dots + 1) % 4
                print(f"\rWaiting for completion{'.' * dots + ' ' * (3-dots)}", end='', flush=True)
                if error[0]:
                    print()  # Clear the waiting line
                    raise error[0]
            
            print()  # Clear the waiting line
            if not completed[0]:
                raise TimeoutError(f"Test timed out after {seconds} seconds")
            
            return result[0]
            
        return wrapper
    return decorator
