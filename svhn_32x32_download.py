import urllib

print("Downloading Test Folder")
urllib.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "data/test_32x32.mat")
print("Test Folder Images Download Done")

print("Downloading Train Folder")
urllib.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "data/train_32x32.mat")
print("Train Folder Images Download Done")

print("Downloading Extra Folder")
urllib.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "data/extra_32x32.mat")
print("Extra Folder Images Download Done")