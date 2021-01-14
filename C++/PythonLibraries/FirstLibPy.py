import ctypes
import pathlib


if __name__=="__main__":
    libname = pathlib.Path().absolute() / "firstlib.so"
    c_lib = ctypes.CDLL("C:/users/avier/Documents/Github/Earth-Science/C++/PythonLibraries/firstlib.so")
