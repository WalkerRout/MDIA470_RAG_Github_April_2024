
# external modules
import os
import tempfile

# internal modules
# N/A

"""
  UserStorage encapsulates a method of storing user-uploaded files

  Current implementation uses tempfile to create and serialize the handle to a temporary
  directory, which is then stored with flask-session for each user. This is not scalable,
  and there are bugs present where the temporary directory is deleted, but the file handle
  is stilled stored in the flask-session cache. 

  Should implement logic to check whether an error occurs when trying to trace handle location, 
  where re-initializing with logic in UserStorage.__init__ should create a new, working location
"""
class UserStorage:
  def __init__(self):
    self.handle = tempfile.TemporaryDirectory(delete=False)
    # tandem lists; can only add/remove files using 
    self.__upload_names = []
    self.__temporary_names = []

  def cleanup(self):
    self.handle.cleanup()
    self.__init__()

  def empty(self) -> bool:
    return len(self.__upload_names) > 0

  def path(self) -> str:
    return self.handle.name

  def add_file(self, upload_file_name, upload_file_bytes):
    # suffix of pdf since PDFLoader looks for .pdf files...
    with tempfile.NamedTemporaryFile(dir=self.path(), delete=False, suffix=".pdf") as tfile:
      tfile.write(upload_file_bytes)
      self.__upload_names.append(upload_file_name)
      self.__temporary_names.append(tfile.name)

  def upload_names(self):
    return self.__upload_names

  def temporary_names(self):
    return self.__temporary_names

  def find_upload_name(self, search: str) -> int:
    return self.__upload_names.index(search)

  def find_temporary_name(self, search: str) -> int:
    return self.__temporary_names.index(search)

  def remove_files_by_index(self, index):
    if index >= len(self.__upload_names):
      return
    file_name = self.__temporary_names[index]
    self.__upload_names.pop(index)
    self.__temporary_names.pop(index)
    os.remove(self.path() + file_name)