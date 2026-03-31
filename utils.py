import hashlib

def file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()