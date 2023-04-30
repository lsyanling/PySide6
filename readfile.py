
import struct
from os import sys
from collections import namedtuple

class fileHeaderStruct():
    def __init__(self, buffer):
        # 下面的 "<II" 是格式字符串 表示按小端字节序读两个unsigned 并将结果保存到两个变量
        # 详见 https://docs.python.org/zh-cn/3/library/struct.html#format-strings
        self.identifier, self.structsize = struct.unpack("<II", buffer)

'''
C++ 中的struct
struct A {
	unsigned a = 1, b = 2;
}
写文件
{
    ofstream of("Oculus_20210728_110209.oculus", ios_base::binary);
    A a;
    of.write((const char*)&a, sizeof(a));
}
'''
filePath = sys.path[0] + "/data/Oculus_20210728_110209.oculus"

file = open(filePath, "rb")

# 读8字节
buffer = file.read(8)

fileHeader = fileHeaderStruct(buffer)

print(fileHeader.identifier)

file.close()
