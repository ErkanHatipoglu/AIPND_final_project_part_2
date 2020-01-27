# this file is for first time setup only
# to download flowers data from udacity
# thanks to Udacity AI Programming with Python Nanodegree Program

# https://stackoverflow.com/questions/19602931/basic-http-file-downloading-and-saving-to-disk-in-python
# thanks to @ https://stackoverflow.com/users/2702249/om-prakash-sao
print('downloading flowers data please wait...')
import urllib.request 
urllib.request.urlretrieve("https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz", "flower_data.tar.gz")
print('download finished!')

# https://stackoverflow.com/questions/48466421/python-how-to-decompress-a-gzip-file-to-an-uncompressed-file-on-disk
# thanks to: @ https://stackoverflow.com/users/532312/rakesh

print('extracting data...')
import tarfile
tar = tarfile.open("flower_data.tar.gz")
tar.extractall("flowers/")
tar.close()
print('finished!')

# https://www.dummies.com/programming/python/how-to-delete-a-file-in-python/
# thanks to: @ https://www.dummies.com/?s=&a=john-paul-mueller

print('removing tar file...')
import os
os.remove("flower_data.tar.gz")
print("File Removed!")