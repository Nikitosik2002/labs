import requests
import os

class HDFSClient:
    def __init__(self, host, port, username):
        self.host = host
        self.port = port
        self.username = username
        self.base_url = f"http://{host}:{port}/webhdfs/v1"

    def mkdir(self, directory):
        url = f"{self.base_url}/{directory}?op=MKDIRS&user.name={self.username}"
        response = requests.put(url)
        if response.status_code == 200:
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Failed to create directory '{directory}'. Error: {response.text}")

    def put(self, local_file, hdfs_file):
        url = f"{self.base_url}/{hdfs_file}?op=CREATE&user.name={self.username}&overwrite=true"
        response = requests.put(url, allow_redirects=False)
        if response.status_code == 307:
            put_url = response.headers['Location']
            with open(local_file, 'rb') as f:
                requests.put(put_url, data=f)
            print(f"File '{local_file}' uploaded successfully to '{hdfs_file}'.")
        else:
            print(f"Failed to upload file '{local_file}' to '{hdfs_file}'. Error: {response.text}")

    def get(self, hdfs_file, local_file):
        url = f"{self.base_url}/{hdfs_file}?op=OPEN&user.name={self.username}"
        response = requests.get(url, allow_redirects=False)
        if response.status_code == 307:
            get_url = response.headers['Location']
            with requests.get(get_url, stream=True) as r:
                with open(local_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"File '{hdfs_file}' downloaded successfully to '{local_file}'.")
        else:
            print(f"Failed to download file '{hdfs_file}' to '{local_file}'. Error: {response.text}")

    def append(self, local_file, hdfs_file):
        url = f"{self.base_url}/{hdfs_file}?op=APPEND&user.name={self.username}"
        response = requests.post(url, allow_redirects=False)
        if response.status_code == 307:
            append_url = response.headers['Location']
            with open(local_file, 'rb') as f:
                requests.post(append_url, data=f)
            print(f"File '{local_file}' appended successfully to '{hdfs_file}'.")
        else:
            print(f"Failed to append file '{local_file}' to '{hdfs_file}'. Error: {response.text}")

    def delete(self, hdfs_file):
        url = f"{self.base_url}/{hdfs_file}?op=DELETE&user.name={self.username}&recursive=true"
        response = requests.delete(url)
        if response.status_code == 200:
            print(f"File '{hdfs_file}' deleted successfully.")
        else:
            print(f"Failed to delete file '{hdfs_file}'. Error: {response.text}")

    def ls(self, directory="."):
        url = f"{self.base_url}/{directory}?op=LISTSTATUS&user.name={self.username}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for file_status in data['FileStatuses']['FileStatus']:
                print(file_status['pathSuffix'])
        else:
            print(f"Failed to list directory '{directory}'. Error: {response.text}")

    def cd(self, directory):
        os.chdir(directory)

    def lls(self, directory="."):
        for item in os.listdir(directory):
            print(item)

    def lcd(self, directory):
        os.chdir(directory)


# Пример использования
if __name__ == "__main__":
    host = "localhost"
    port = 50070
    username = "aslebedev"
    client = HDFSClient(host, port, username)

    client.mkdir("testdir")
    client.put("localfile.txt", "testdir/hdfsfile.txt")
    client.get("testdir/hdfsfile.txt", "localfile_copy.txt")
    client.append("localfile.txt", "testdir/hdfsfile.txt")
    client.ls("testdir")
    client.delete("testdir/hdfsfile.txt")
    client.ls("testdir")
