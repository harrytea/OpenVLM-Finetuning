import os
import shutil

def delete_pycache_dirs(root_dir):
    """
    递归删除指定目录及其子目录下的所有 '__pycache__' 文件夹。
    :param root_dir: 要清理的根目录
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录中的所有文件夹
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"已删除: {pycache_path}")
            except Exception as e:
                print(f"无法删除 {pycache_path}：{e}")

if __name__ == "__main__":
    # 获取当前工作目录
    current_directory = os.getcwd()
    print(f"正在清理目录: {current_directory}")
    delete_pycache_dirs(current_directory)
    print("清理完成！")
