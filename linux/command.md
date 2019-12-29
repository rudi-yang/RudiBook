# 常用命令

https://blog.csdn.net/zhangliao613/article/details/79021606


查看CPU个数:
 cat /proc/cpuinfo | grep "physical id" | uniq | wc -l

查看CPU核数:
 cat /proc/cpuinfo | grep "cpu cores" | uniq

查看CPU型号:
 cat /proc/cpuinfo | grep 'model name' |uniq

查看内存总数:
    cat /proc/meminfo | grep MemTotal

    MemTotal: 32941268 kB //内存32G

查看硬盘空间大小:
df  -k i-h