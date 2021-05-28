import pickle
import matplotlib

pickle_file = open("record_mean.dat", "rb")
objects = []
while True:
    try:
        objects.append(pickle.load(pickle_file))
    except EOFError:
        break

pickle_file.close()
