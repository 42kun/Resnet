import os
import sys
import zipfile

from urllib.request import urlretrieve

def download_file(download_url,dest_dir):
    # 检查是否存在dest_dir目录,如果不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # 根据下载url来获取下载的文件名
    file_name = download_url.split("/")[-1]
    # 生成文件存放目录
    file_path = os.path.join(dest_dir, file_name)

    # 如果文件不存在，则开始下载过程
    if not os.path.exists(file_path):
        # 回调函数
        def callBack(count, block_size, total_size):
            # 进度条
            # 可以考虑用tqdm
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (file_name, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()

        # 下载主函数
        __, _ = urlretrieve(download_url, file_path, callBack)
        print()
        # 获取文件信息
        statinfo = os.stat(file_path)
        print('Successfully downloaded', file_name, statinfo.st_size, 'bytes.')
         #解压zip文件的标准流程，file_path为压缩文件目录，dest_dir指定解压后文件放置在哪个目录
        # 如果下载的文件不需要解压，请忽略下面的语句
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(dest_dir)

if __name__ == "__main__":
    download_file("http://42kun.cn/wp-content/uploads/2019/09/data_batch_1.zip",
                  "download")