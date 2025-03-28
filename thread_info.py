import threading

def print_thread_info():
    print("Active threads:")
    for thread in threading.enumerate():
        print(f"Thread name: {thread.name}, Thread ID: {thread.ident}")

# 在程序的合适位置调用
print_thread_info()