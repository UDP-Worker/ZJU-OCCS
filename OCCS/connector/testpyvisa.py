import pyvisa as visa

# 显示所有的已连接设备，方便后续连接
rm= visa.ResourceManager('/opt/keysight/iolibs/libktvisa32.so')
list_instr = rm.list_resources()
print(list_instr)
